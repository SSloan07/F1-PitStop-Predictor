"""Recopila vueltas y clima por carrera; repetir el script reanuda la descarga."""
import time
from pathlib import Path

import fastf1
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print

BASE = Path(__file__).resolve().parent
carpeta_cache = BASE / 'cache_fastf1_carreras'
# Separada de los archivos anteriores, que no incluyen clima.
carpeta_vueltas = BASE / 'datos_carreras_completos_2022_2025'
carpeta_cache.mkdir(exist_ok=True)
carpeta_vueltas.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(carpeta_cache))


def guardar_atomico(tabla, archivo):
    temporal = archivo.with_suffix('.tmp')
    tabla.to_pickle(temporal)
    temporal.replace(archivo)


def resumir_telemetria(laps):
    """Resume las muestras de cada vuelta sin cambiar la granularidad del dataset."""
    resumen = []
    for _, lap in laps.iterlaps():
        datos = lap.get_car_data()
        fila = {'Telemetry_Samples': len(datos)}
        for columna in ('Speed', 'RPM', 'Throttle', 'Brake', 'nGear', 'DRS'):
            valores = pd.to_numeric(datos[columna], errors='coerce') if columna in datos else pd.Series(dtype=float)
            fila[f'Telemetry_{columna}_mean'] = valores.mean()
            fila[f'Telemetry_{columna}_max'] = valores.max()
        # 104 indica datos de acelerador no disponibles según FastF1.
        acelerador = pd.to_numeric(datos['Throttle'], errors='coerce') if 'Throttle' in datos else pd.Series(dtype=float)
        validos = acelerador.between(0, 100)
        fila['Telemetry_Throttle_valid_samples'] = int(validos.sum())
        fila['Telemetry_Throttle_valid_fraction'] = validos.sum() / len(datos) if len(datos) else float('nan')
        fila['Telemetry_Throttle_valid_mean'] = acelerador[validos].mean()
        fila['Telemetry_Throttle_valid_max'] = acelerador[validos].max()
        resumen.append(fila)
    return pd.DataFrame(resumen)


def es_limite_api(error):
    """Reconoce errores de límite de FastF1 y respuestas HTTP 429."""
    visitados = set()
    while error is not None and id(error) not in visitados:
        visitados.add(id(error))
        respuesta = getattr(error, 'response', None)
        texto = f'{type(error).__name__} {error}'.lower()
        if (getattr(respuesta, 'status_code', None) == 429 or
                'ratelimit' in texto.replace(' ', '').replace('_', '') or
                'rate limit' in texto or 'too many requests' in texto or '429' in texto):
            return True
        error = error.__cause__ or error.__context__
    return False


anios = range(2022, 2026)
tablas_carreras = []
registro_cargas = []
inicio = time.monotonic()
limite_alcanzado = False

for anio in anios:
    try:
        calendario = fastf1.get_event_schedule(anio, include_testing=False)
    except Exception as error:
        registro_cargas.append({'Temporada': anio, 'Ronda': None, 'GranPremio': None,
                                'Estado': 'Error de calendario', 'Detalle': str(error)})
        print(f'No se pudo cargar el calendario de {anio}: {error}')
        if es_limite_api(error):
            limite_alcanzado = True
            break
        continue

    columnas_sesion = [f'Session{i}' for i in range(1, 6)]
    carreras = calendario.loc[calendario[columnas_sesion].eq('Race').any(axis=1)]
    for _, evento in carreras.iterrows():
        ronda = int(evento['RoundNumber'])
        nombre = str(evento['EventName'])
        archivo = carpeta_vueltas / f'{anio}_{ronda:02d}.pkl'
        try:
            if archivo.exists():
                vueltas = pd.read_pickle(archivo)
                vueltas = vueltas.drop(columns=['PitEnProximas3Vueltas'], errors='ignore')
                estado = 'Recuperada'
            else:
                print(f'Cargando {anio}, ronda {ronda}: {nombre}...')
                carrera = fastf1.get_session(anio, ronda, 'R')
                carrera.load(laps=True, telemetry=True, weather=True, messages=True)
                if carrera.laps.empty:
                    raise ValueError('La sesión no contiene vueltas.')
                if carrera.weather_data.empty:
                    raise ValueError('La sesión no contiene datos meteorológicos; se reintentará.')
                # FastF1 selecciona la observación de clima correspondiente a cada vuelta.
                clima = carrera.laps.get_weather_data().reset_index(drop=True)
                vueltas = pd.DataFrame(carrera.laps).reset_index(drop=True).copy()
                if len(clima) != len(vueltas):
                    raise ValueError('El clima no coincide con el número de vueltas.')
                clima = clima.add_prefix('Weather_')
                telemetria = resumir_telemetria(carrera.laps)
                if telemetria['Telemetry_Samples'].sum() == 0:
                    raise ValueError('La sesión no contiene telemetría; se reintentará.')
                # TrackStatus contiene todos los códigos observados durante la vuelta.
                estados = vueltas['TrackStatus'].astype('string')
                vueltas['SafetyCar'] = estados.str.contains('4', regex=False).astype('boolean')
                vueltas['VirtualSafetyCar'] = (estados.str.contains('6', regex=False) | estados.str.contains('7', regex=False)).astype('boolean')
                vueltas = pd.concat([vueltas, clima, telemetria], axis=1)
                vueltas.insert(0, 'Temporada', anio)
                vueltas.insert(1, 'Ronda', ronda)
                vueltas.insert(2, 'GranPremio', nombre)
                guardar_atomico(vueltas, archivo)
                estado = 'Descargada'
            tablas_carreras.append(vueltas)
            registro_cargas.append({'Temporada': anio, 'Ronda': ronda, 'GranPremio': nombre,
                                    'Estado': estado, 'Detalle': f'{len(vueltas)} vueltas'})
        except Exception as error:
            print(f'No se pudo procesar {anio} - {nombre}: {error}')
            registro_cargas.append({'Temporada': anio, 'Ronda': ronda, 'GranPremio': nombre,
                                    'Estado': 'Error de carrera', 'Detalle': str(error)})
            if es_limite_api(error):
                limite_alcanzado = True
                break
    if limite_alcanzado:
        break

resumen_cargas = pd.DataFrame(registro_cargas)
if limite_alcanzado:
    print('Límite de la API alcanzado. Espera a que se restablezca y vuelve a ejecutar el extractor.')
if not tablas_carreras:
    display(resumen_cargas)
    raise RuntimeError('No se pudo cargar ninguna carrera en esta ejecución.')

df_carreras = pd.concat(tablas_carreras, ignore_index=True, sort=False)
guardar_atomico(df_carreras, carpeta_vueltas / 'vueltas_2022_2025.pkl')
print(f'Vueltas reunidas en esta ejecución: {len(df_carreras):,} | Carreras: {len(tablas_carreras)} | '
      f'Tiempo: {(time.monotonic() - inicio) / 60:.1f} minutos')
print('Resumen por temporada:')
display(df_carreras.groupby('Temporada').agg(
    carreras=('Ronda', 'nunique'), vueltas=('LapNumber', 'size'),
    pilotos=('Driver', 'nunique')))
print('Carreras o calendarios con errores:')
display(resumen_cargas[resumen_cargas['Estado'].str.startswith('Error')])
print('Primeras vueltas:')
display(df_carreras.head())
