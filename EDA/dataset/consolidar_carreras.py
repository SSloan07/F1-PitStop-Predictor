import time
from pathlib import Path
import fastf1
import pandas as pd
from IPython.display import display

# Las descargas se guardan por carrera para poder retomar el proceso si se interrumpe.
# La caché de FastF1 se mantiene separada de los DataFrames procesados.
carpeta_cache = Path('cache_fastf1_carreras')
carpeta_vueltas = Path('datos_carreras_2022_2025')
carpeta_cache.mkdir(exist_ok=True)
carpeta_vueltas.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(carpeta_cache))

anios = range(2022, 2026)
tablas_carreras = []
registro_cargas = []
inicio = time.monotonic()

for anio in anios:
    try:
        calendario = fastf1.get_event_schedule(anio, include_testing=False)
    except Exception as error:
        registro_cargas.append({'Temporada': anio, 'Ronda': None, 'GranPremio': None,
                                'Estado': 'Error de calendario', 'Detalle': str(error)})
        print(f'No se pudo cargar el calendario de {anio}: {error}')
        continue

    # Solo eventos cuyo calendario contiene una sesión llamada Race; no incluye sprints.
    columnas_sesion = [f'Session{i}' for i in range(1, 6)]
    carreras = calendario.loc[calendario[columnas_sesion].eq('Race').any(axis=1)]
    for _, evento in carreras.iterrows():
        ronda = int(evento['RoundNumber'])
        nombre = str(evento['EventName'])
        archivo = carpeta_vueltas / f'{anio}_{ronda:02d}.pkl'
        try:
            if archivo.exists():
                vueltas = pd.read_pickle(archivo)
                estado = 'Recuperada'
            else:
                print(f'Cargando {anio}, ronda {ronda}: {nombre}...')
                carrera = fastf1.get_session(anio, ronda, 'R')
                # La tabla de vueltas basta para este primer análisis; no descarga telemetría.
                carrera.load(laps=True, telemetry=False, weather=False, messages=True)
                vueltas = pd.DataFrame(carrera.laps).copy()
                if vueltas.empty:
                    raise ValueError('La sesión no contiene vueltas.')
                vueltas.insert(0, 'Temporada', anio)
                vueltas.insert(1, 'Ronda', ronda)
                vueltas.insert(2, 'GranPremio', nombre)
                vueltas.to_pickle(archivo)
                estado = 'Descargada'
            tablas_carreras.append(vueltas)
            registro_cargas.append({'Temporada': anio, 'Ronda': ronda, 'GranPremio': nombre,
                                    'Estado': estado, 'Detalle': f'{len(vueltas)} vueltas'})
        except Exception as error:
            print(f'No se pudo procesar {anio} - {nombre}: {error}')
            registro_cargas.append({'Temporada': anio, 'Ronda': ronda, 'GranPremio': nombre,
                                    'Estado': 'Error de carrera', 'Detalle': str(error)})

resumen_cargas = pd.DataFrame(registro_cargas)
if not tablas_carreras:
    display(resumen_cargas)
    raise RuntimeError('No se pudo cargar ninguna carrera. Revisa el registro anterior.')

df_carreras = pd.concat(tablas_carreras, ignore_index=True, sort=False)
df_carreras.to_pickle(carpeta_vueltas / 'vueltas_2022_2025.pkl')
print(f'Vueltas reunidas: {len(df_carreras):,} | Carreras: {len(tablas_carreras)} | '
      f'Tiempo: {(time.monotonic() - inicio) / 60:.1f} minutos')
print('Resumen por temporada:')
display(df_carreras.groupby('Temporada').agg(
    carreras=('Ronda', 'nunique'), vueltas=('LapNumber', 'size'),
    pilotos=('Driver', 'nunique')))
print('Carreras o calendarios con errores:')
display(resumen_cargas[resumen_cargas['Estado'].str.startswith('Error')])
print('Primeras vueltas:')
display(df_carreras.head())
