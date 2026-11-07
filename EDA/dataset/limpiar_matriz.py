"""Limpieza inicial de la matriz histórica; no prepara entradas de modelos."""
import hashlib
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent / 'datos_carreras_completos_2022_2025'
ORIGEN = BASE / 'vueltas_2022_2025_compound_reconstruido.pkl'
SALIDA = BASE / 'vueltas_2022_2025_limpieza_v1.pkl'
INFORME = BASE / 'limpieza_v1_informe.json'
COBERTURA = BASE / 'limpieza_v1_cobertura.csv'
CLAVES = ['Temporada', 'Ronda', 'DriverNumber', 'LapNumber']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if any(path.exists() for path in (SALIDA, INFORME, COBERTURA)):
        raise FileExistsError('La versión v1 ya existe; conserve sus artefactos y use una nueva versión.')
    matriz = pd.read_pickle(ORIGEN)
    if matriz[CLAVES].isna().any().any() or matriz.duplicated(CLAVES).any():
        raise ValueError('Claves incompletas o duplicadas: no se produjo salida.')
    limpia = matriz.copy()
    texto = limpia['Compound'].astype('string').str.strip()
    ausente = texto.isna() | texto.str.lower().isin(['none', 'nan', ''])
    limpia.loc[ausente, 'Compound'] = pd.NA
    grupo = ['Temporada', 'Ronda', 'GranPremio', 'DriverNumber']
    ordenada = limpia.sort_values(grupo + ['LapNumber']).copy()
    ordenada['salto_vuelta'] = ordenada.groupby(grupo, dropna=False)['LapNumber'].diff().gt(1)
    ordenada['compound_ausente'] = ordenada['Compound'].isna()
    ordenada['tyrelife_ausente'] = ordenada['TyreLife'].isna()
    cobertura = ordenada.groupby(grupo, dropna=False).agg(
        filas=('LapNumber', 'size'), compound_ausente=('compound_ausente', 'sum'),
        tyrelife_ausente=('tyrelife_ausente', 'sum'), saltos=('salto_vuelta', 'sum'),
        vueltas_generadas=('FastF1Generated', 'sum')).reset_index()
    limpia.to_pickle(SALIDA)
    cobertura.to_csv(COBERTURA, index=False)
    informe = {
        'version': 'limpieza_v1', 'origen': str(ORIGEN.name), 'sha256_origen': sha256(ORIGEN),
        'salida': str(SALIDA.name), 'sha256_salida': sha256(SALIDA),
        'filas': len(limpia), 'columnas': len(limpia.columns),
        'compound_marcadores_normalizados': int((ausente & matriz['Compound'].notna()).sum()),
        'compound_ausente': int(limpia['Compound'].isna().sum()),
        'tyrelife_ausente': int(limpia['TyreLife'].isna().sum()),
        'saltos_entre_vueltas': int(ordenada['salto_vuelta'].sum()),
        'advertencia': 'Matriz histórica: no valida etiquetas ni disponibilidad temporal para entrenamiento.',
    }
    INFORME.write_text(json.dumps(informe, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(informe, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
