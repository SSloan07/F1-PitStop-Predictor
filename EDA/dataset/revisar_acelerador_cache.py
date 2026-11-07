"""Recalcula calidad del acelerador desde la caché usando SessionTime de FastF1."""
from pathlib import Path

import fastf1
import pandas as pd

BASE = Path(__file__).resolve().parent
CACHE = BASE / 'cache_fastf1_carreras'
DATOS = BASE / 'datos_carreras_completos_2022_2025'
COLUMNAS = ['Telemetry_Throttle_valid_samples', 'Telemetry_Throttle_valid_fraction',
            'Telemetry_Throttle_valid_mean', 'Telemetry_Throttle_valid_max']


def actualizar(archivo):
    vueltas = pd.read_pickle(archivo)
    temporada, ronda = (int(parte) for parte in archivo.stem.split('_'))
    sesion = fastf1.get_session(temporada, ronda, 'R')
    sesion.load(laps=True, telemetry=True, weather=False, messages=False)
    for datos in sesion.car_data.values():
        if not datos['SessionTime'].is_monotonic_increasing:
            raise ValueError('La telemetría no está ordenada por SessionTime')
    filas = []
    for fila in vueltas.itertuples(index=False):
        datos = sesion.car_data.get(str(fila.DriverNumber))
        if datos is None or pd.isna(fila.LapStartTime) or pd.isna(fila.Time):
            filas.append((0, float('nan'), float('nan'), float('nan')))
            continue
        inicio = datos['SessionTime'].searchsorted(fila.LapStartTime, side='left')
        fin = datos['SessionTime'].searchsorted(fila.Time, side='right')
        valores = pd.to_numeric(datos['Throttle'].iloc[inicio:fin], errors='coerce') if 'Throttle' in datos else pd.Series(dtype=float)
        validos = valores.between(0, 100)
        cantidad = int(validos.sum())
        filas.append((cantidad, cantidad / len(valores) if len(valores) else float('nan'),
                      valores[validos].mean(), valores[validos].max()))
    for columna, valores in zip(COLUMNAS, zip(*filas)):
        vueltas[columna] = list(valores)
    temporal = archivo.with_suffix('.tmp')
    vueltas.to_pickle(temporal)
    temporal.replace(archivo)


def main():
    fastf1.Cache.enable_cache(str(CACHE))
    fastf1.Cache.offline_mode(True)
    errores = []
    archivos = sorted(DATOS.glob('20??_??.pkl'))
    for archivo in archivos:
        try:
            actualizar(archivo)
            print(f'{archivo.name}: actualizada', flush=True)
        except Exception as error:
            errores.append(archivo.name)
            print(f'{archivo.name}: ERROR {error}', flush=True)
    if errores:
        raise SystemExit(f'No se actualizaron: {", ".join(errores)}. No se regeneró el consolidado.')
    consolidado = pd.concat([pd.read_pickle(p) for p in archivos], ignore_index=True, sort=False)
    archivo = DATOS / 'vueltas_2022_2025.pkl'
    temporal = archivo.with_suffix('.tmp')
    consolidado.to_pickle(temporal)
    temporal.replace(archivo)
    print(f'Consolidado regenerado: {len(archivos)} carreras, {len(consolidado)} vueltas.')


if __name__ == '__main__':
    main()
