"""Compare the consolidated source matrix with the tyre-reconstructed candidate."""
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent / 'datos_carreras_completos_2022_2025'
ORIGINAL = BASE / 'vueltas_2022_2025.pkl'
CANDIDATE = BASE / 'vueltas_2022_2025_compound_reconstruido.pkl'
AUDIT = BASE / 'auditoria_reconstruccion_compound.csv'
MISSING_LABELS = {'', 'NONE', 'NAN', 'UNKNOWN', '<NA>'}
KEY = ['Temporada', 'Ronda', 'DriverNumber', 'LapNumber']


def missing_values(series):
    normalized = series.astype('string').str.strip()
    return normalized.isna() | normalized.str.upper().isin(MISSING_LABELS)


def main():
    original = pd.read_pickle(ORIGINAL)
    candidate = pd.read_pickle(CANDIDATE)
    audit = pd.read_csv(AUDIT)

    if original.shape != candidate.shape or not original.columns.equals(candidate.columns):
        raise AssertionError('La candidata debe conservar filas, columnas y su orden.')
    non_compound_columns = [column for column in original.columns if column != 'Compound']
    changed_columns = [
        column for column in original.columns
        if not original[column].equals(candidate[column])
    ]
    if changed_columns != ['Compound']:
        raise AssertionError(f'Se esperaba modificar solo Compound; cambiaron {changed_columns}.')
    if not original[non_compound_columns].equals(candidate[non_compound_columns]):
        raise AssertionError('Hay cambios fuera de Compound.')

    eligible = (
        original['Stint'].notna()
        & original['LapNumber'].notna()
        & ~original['FastF1Generated'].fillna(False).astype(bool)
    )
    compound_before = missing_values(original['Compound'])
    compound_after = missing_values(candidate['Compound'])
    tyre_before = missing_values(original['TyreLife'])
    tyre_after = missing_values(candidate['TyreLife'])
    changed_compound = original['Compound'].astype('string') != candidate['Compound'].astype('string')

    if not (changed_compound <= compound_before).all():
        raise AssertionError('Se sobrescribió un compuesto que no estaba marcado como ausente.')
    if (compound_after & changed_compound).any():
        raise AssertionError('Una fila marcada como reconstruida continúa ausente.')
    if not original['TyreLife'].equals(candidate['TyreLife']):
        raise AssertionError('TyreLife debe permanecer intacto en esta reconstrucción.')

    duplicate_count = int(original.loc[original['LapNumber'].notna()].duplicated(KEY).sum())
    if duplicate_count:
        raise AssertionError(f'Hay {duplicate_count} claves piloto-vuelta duplicadas.')

    print(f'Filas: {len(original):,} | columnas: {len(original.columns)}')
    print(f'Filas elegibles para reconstrucción: {int(eligible.sum()):,}')
    print(f'Claves de vuelta duplicadas: {duplicate_count}')
    print(f'Columnas modificadas: {changed_columns}')
    print(f'Filas con Compound reconstruido: {int(changed_compound.sum())}')
    print(f'Stints auditados: {len(audit)} | reconstruidos: {(audit.Estado == "reconstructed").sum()}')
    print(f'Filas reconstruidas en auditoría: {int(audit.loc[audit.Estado == "reconstructed", "Filas_afectadas"].sum())}')
    print(f'Tyrelife idéntico entre matrices: {original["TyreLife"].equals(candidate["TyreLife"])}')
    print('\nCobertura elegible por temporada (faltantes originales -> candidatos):')
    for season in sorted(original['Temporada'].dropna().unique()):
        season_rows = original['Temporada'] == season
        season_eligible = eligible & season_rows
        print(
            f'{int(season)}: '
            f'Compound {int((compound_before & season_eligible).sum())} -> {int((compound_after & season_eligible).sum())}; '
            f'TyreLife {int((tyre_before & season_eligible).sum())} -> {int((tyre_after & season_eligible).sum())}'
        )

    remaining = candidate.loc[compound_after & eligible]
    print('\nCompound elegible que permanece ausente:')
    if remaining.empty:
        print('ninguno')
    else:
        print(remaining.groupby(['Temporada', 'Ronda', 'GranPremio']).size().to_string())

    tokens = original.loc[compound_before, 'Compound'].astype('string').value_counts(dropna=False)
    print('\nMarcadores textuales de Compound ausente en la matriz original:')
    print(tokens.to_string())


if __name__ == '__main__':
    main()
