"""Build a separate canonical matrix candidate using validated tyre-cache data.

The original consolidated matrix and FastF1 cache are never modified. A stint is
filled only when its timing-app compound is unique and that event's mapping from
FastF1's zero-based timing-app Stint to the matrix's one-based Stint has at least
10 comparable known stints with 100% agreement.
"""
from pathlib import Path
import pickle
from functools import lru_cache

import pandas as pd


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / 'datos_carreras_completos_2022_2025'
CACHE_DIR = BASE / 'cache_fastf1_carreras'
INPUT = DATA_DIR / 'vueltas_2022_2025.pkl'
OUTPUT = DATA_DIR / 'vueltas_2022_2025_compound_reconstruido.pkl'
AUDIT = DATA_DIR / 'auditoria_reconstruccion_compound.csv'
MIN_COMPARABLE_STINTS = 10

GROUP = ['Temporada', 'Ronda', 'GranPremio', 'DriverNumber', 'Stint']
KNOWN_MISSING = {'', 'NONE', 'NAN', 'UNKNOWN', '<NA>'}


def normalized_compound(values):
    """Return non-null, non-placeholder source compound labels."""
    labels = values.dropna().astype(str).str.strip()
    return labels[~labels.str.upper().isin(KNOWN_MISSING)]


@lru_cache(maxsize=None)
def timing_cache(year, grand_prix):
    """Find and read the race's local timing-app cache without using the API."""
    event_token = grand_prix.replace(' ', '_')
    matches = list((CACHE_DIR / str(int(year))).glob(f'*{event_token}*/*_Race/timing_app_data.ff1pkl'))
    if len(matches) != 1:
        return None, None
    path = matches[0]
    with path.open('rb') as source_file:
        payload = pickle.load(source_file)
    return payload.get('data'), path


def main():
    matrix = pd.read_pickle(INPUT)
    result = matrix.copy()
    normalized = matrix['Compound'].astype('string').str.strip()
    missing = normalized.isna() | normalized.str.upper().isin(KNOWN_MISSING)
    base_valid = (
        matrix['Stint'].notna()
        & matrix['LapNumber'].notna()
        & ~matrix['FastF1Generated'].fillna(False).astype(bool)
    )

    group_records = []
    for key, indexes in matrix.loc[base_valid].groupby(GROUP, sort=False, dropna=False).groups.items():
        year, round_number, grand_prix, driver_number, stint_value = key
        timing, cache_path = timing_cache(year, grand_prix)
        if timing is None or not {'Driver', 'Stint', 'Compound'}.issubset(timing.columns):
            source_labels = []
            cache_path_text = '' if cache_path is None else str(cache_path)
        else:
            driver_rows = timing[timing['Driver'].astype(str) == str(driver_number)]
            source_stint = int(stint_value) - 1
            source_rows = driver_rows[driver_rows['Stint'] == source_stint]
            source_labels = sorted(normalized_compound(source_rows['Compound']).unique().tolist())
            cache_path_text = str(cache_path.relative_to(BASE.parent))

        known_matrix_labels = sorted(normalized_compound(matrix.loc[indexes, 'Compound']).unique().tolist())
        missing_count = int(missing.loc[indexes].sum())
        group_records.append({
            'key': key,
            'indexes': indexes,
            'year': int(year),
            'round': int(round_number),
            'grand_prix': str(grand_prix),
            'driver': str(driver_number),
            'stint': int(stint_value),
            'known_matrix': known_matrix_labels,
            'source_labels': source_labels,
            'missing_count': missing_count,
            'cache_path': cache_path_text,
        })

    # Establish the stint-number mapping independently for each event using
    # known matrix compounds. Ambiguous source records are excluded.
    event_validation = {}
    for item in group_records:
        if not item['known_matrix'] or len(item['source_labels']) != 1:
            continue
        event_key = (item['year'], item['round'], item['grand_prix'])
        stats = event_validation.setdefault(event_key, {'comparable': 0, 'matches': 0})
        stats['comparable'] += 1
        stats['matches'] += int(item['known_matrix'] == item['source_labels'])

    audit_rows = []
    filled_groups = 0
    filled_rows = 0
    for item in group_records:
        if item['missing_count'] == 0:
            continue
        event_key = (item['year'], item['round'], item['grand_prix'])
        validation = event_validation.get(event_key, {'comparable': 0, 'matches': 0})
        event_mapping_valid = (
            validation['comparable'] >= MIN_COMPARABLE_STINTS
            and validation['matches'] == validation['comparable']
        )
        if len(item['source_labels']) != 1:
            status = 'ambiguous_or_absent_source_compound'
            chosen = ''
        elif not event_mapping_valid:
            status = 'event_mapping_not_validated'
            chosen = ''
        else:
            status = 'reconstructed'
            chosen = item['source_labels'][0]
            result.loc[item['indexes'], 'Compound'] = chosen
            filled_groups += 1
            filled_rows += item['missing_count']
        audit_rows.append({
            'Temporada': item['year'],
            'Ronda': item['round'],
            'GranPremio': item['grand_prix'],
            'DriverNumber': item['driver'],
            'Stint_matrix': item['stint'],
            'Filas_afectadas': item['missing_count'],
            'Compuesto_matriz_observado': '|'.join(item['known_matrix']),
            'Compuesto_timing_app': '|'.join(item['source_labels']),
            'Compuesto_asignado': chosen,
            'Stints_comparables_evento': validation['comparable'],
            'Coincidencias_evento': validation['matches'],
            'Estado': status,
            'Cache_fuente': item['cache_path'],
        })

    # Structural guarantees: one output row per input row; only Compound changes.
    if len(result) != len(matrix) or not result.index.equals(matrix.index):
        raise RuntimeError('La matriz candidata cambió el número o el orden de filas.')
    unchanged_columns = [column for column in matrix.columns if column != 'Compound']
    if not result[unchanged_columns].equals(matrix[unchanged_columns]):
        raise RuntimeError('Se modificó una columna distinta de Compound.')

    temporary_output = OUTPUT.with_suffix('.tmp')
    result.to_pickle(temporary_output)
    temporary_output.replace(OUTPUT)
    pd.DataFrame(audit_rows).to_csv(AUDIT, index=False)

    print(f'Filas originales/candidatas: {len(matrix):,}/{len(result):,}')
    print(f'Stints reconstruidos: {filled_groups} | filas con Compound reconstruido: {filled_rows}')
    print(f'Faltantes elegibles restantes: {int((missing & base_valid).sum()) - filled_rows}')
    print(f'Matriz candidata: {OUTPUT}')
    print(f'Auditoría: {AUDIT}')
    print('\nValidación de correspondencia por carrera:')
    for (year, round_number, grand_prix), stats in sorted(event_validation.items()):
        print(f"{year} R{round_number} {grand_prix}: {stats['matches']}/{stats['comparable']}")
    print('\nResultado por stint faltante:')
    print(pd.DataFrame(audit_rows)[['Temporada', 'GranPremio', 'DriverNumber', 'Stint_matrix', 'Compuesto_asignado', 'Estado']].to_string(index=False))


if __name__ == '__main__':
    main()
