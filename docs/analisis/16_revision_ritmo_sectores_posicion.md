# Revisión 3: ritmo, sectores, velocidades de timing y posición

**Fuente local:** `EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_limpieza_v1.pkl`, corte 2022–2023 de 47.997 filas. **Alcance:** 13 columnas del [inventario](13_inventario_columnas_para_modelado.md). La [documentación primaria de FastF1](https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/timing_data.rst) define tiempos, puntos de velocidad y `Position`; su [código fuente](https://raw.githubusercontent.com/theOehrly/Fast-F1/main/fastf1/core.py) muestra transformaciones internas relevantes. Al implementar se fijará la versión exacta de FastF1 usada por el extractor y se comprobará si su comportamiento coincide con el código consultado.

**Estado:** análisis para socializar. No modifica datos ni aprueba la matriz para entrenamiento. Todas las propuestas respetan la predicción al inicio de la vuelta `N`: un valor calculado durante o al final de `N` no entra en `X_N`.

## Cobertura y consistencia

| Columna/familia | Faltantes observados | Control importante |
|---|---:|---|
| `LapTime` | 270 | 79 filas generadas y 191 no generadas. Ningún tiempo presente es no positivo. |
| `Sector1Time` | 1.041 | 79 generadas y 962 no generadas. Es el sector con mayor ausencia. |
| `Sector2Time` | 89 | 79 generadas y 10 no generadas. |
| `Sector3Time` | 178 | 79 generadas y 99 no generadas. |
| `Sector1SessionTime` | 1.147 | Marca temporal de sector 1; más ausencias que `Sector1Time`. |
| `Sector2SessionTime` | 89 | Marca temporal de sector 2. |
| `Sector3SessionTime` | 178 | Marca temporal de sector 3. |
| `SpeedI1`, `SpeedI2` | 80, 79 | Respectivamente, una y cero ausencias no generadas. |
| `SpeedFL` | 1.857 | 1.778 filas con `PitInTime` y 79 generadas. |
| `SpeedST` | 113 | 79 generadas y 34 no generadas. |
| `IsPersonalBest` | 64 | Valor oficial retrospectivo; no coincide necesariamente con el menor `LapTime` bruto si hubo anulación. |
| `Position` | 79 | Todas corresponden a filas generadas; valores presentes entre 1 y 20. |

En las **46.867 filas** con `LapTime` y los tres sectores presentes, la suma de los sectores coincide **exactamente** con `LapTime` en la precisión almacenada. Esto es un control de integridad, no prueba que cada medición hubiera llegado al inicio de la vuelta siguiente. Los sectores pueden aportar **dónde** cambió el ritmo, mientras que `LapTime` aporta el total; esa redundancia funcional se evaluará por ablación en train, sin quitar columnas solo por correlación matemática.

Hay **858 filas** con `LapTime`, `Sector2Time` y `Sector3Time` presentes pero `Sector1Time` ausente. Su diferencia algebraica es positiva en todas, pero tres superan 100 segundos y aparecen en la primera vuelta de Mónaco 2022 bajo circunstancias de boxes/lluvia. También hay **una fila** con los tres sectores presentes y `LapTime` ausente. Son candidatos a reconstrucción determinista **después** de cerrar la vuelta, conservando procedencia y comprobando casos excepcionales; no se rellenarán sin revisión ni se utilizarán en la decisión al inicio de esa misma vuelta. La matriz canónica seguirá teniendo una sola fila y un valor por campo.

Las marcas `Sector1/2/3SessionTime` indican el tiempo de sesión en que se estableció cada sector, según FastF1; no son una hora de recepción. Sirven para ordenar y validar eventos. En el histórico, `Sector3SessionTime` y `Time` difieren en la mayoría de las filas por pequeñas cantidades: la diferencia mediana `Time − Sector3SessionTime` es −0,029 segundos. Por tanto, una prueba de coherencia temporal debe usar tolerancias justificadas, no igualdad exacta de timestamps. Estas tres marcas no entran como magnitudes directas en el experimento base.

## Velocidades: faltantes estructurales y valores procesados

FastF1 define `SpeedI1` y `SpeedI2` como puntos de velocidad de los sectores 1 y 2, `SpeedFL` como el punto de meta y `SpeedST` como un punto rápido dependiente del circuito, todos en km/h. Su documentación advierte que pueden faltar bajo bandera roja y que `SpeedFL` falta en vueltas de entrada a boxes. El [código de FastF1](https://raw.githubusercontent.com/theOehrly/Fast-F1/main/fastf1/core.py) también aplica *forward fill* por piloto a velocidades faltantes en ciertas vueltas originales; para `SpeedFL` excluye las vueltas con `PitInTime`. Por ello, un valor presente de la matriz puede ser un valor propagado y no una medición nueva de esa vuelta.

En este corte, **1.778 de 1.779** filas con `PitInTime` tienen `SpeedFL` ausente; la excepción es Zhou en São Paulo 2022, vuelta 16. Las otras 79 ausencias de `SpeedFL` son filas generadas. Esto hace que tanto `SpeedFL` como su máscara de ausencia en la **vuelta actual** sean un sustituto casi directo de `PitInTime` y produzcan fuga del objetivo. El constructor debe impedirlo de forma explícita. En una vuelta histórica ya completada, la ausencia refleja un hecho pasado; su uso como contexto de la RNN exige aun así una regla de disponibilidad y una comparación de sensibilidad para no atribuir al modelo una señal física de velocidad inexistente.

No se imputará una velocidad faltante con cero. Tampoco se interpretará automáticamente un valor repetido como velocidad observada en dos vueltas: hace falta distinguir la medición original de la propagación que pudo aplicar la biblioteca. Si esa procedencia no puede recuperarse, se documentará el canal como velocidad de timing procesada por FastF1 y se comparará un experimento sin él.

## `Position` y `IsPersonalBest`

La [documentación de FastF1](https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/timing_data.rst) define `Position` como la posición **al final** de la vuelta, no al inicio. Su [implementación](https://raw.githubusercontent.com/theOehrly/Fast-F1/main/fastf1/core.py) ordena por `Time` las filas de todos los pilotos con el mismo `LapNumber` para asignar la posición. Por ello, la posición consolidada es retrospectiva: incluso desplazarla una vuelta para el piloto no demuestra por sí solo que se conociera exactamente al cruzar la línea siguiente, pues pueden faltar tiempos de otros pilotos en ese instante. Se propone dejar el `Position` de la matriz fuera del `X` base hasta construir una posición con feed y timestamp de disponibilidad comprobables; entonces podría entrar como estado dinámico.

`IsPersonalBest` indica el mejor tiempo oficial del piloto y puede depender de la validación de vueltas, por ejemplo tras límites de pista, según la documentación de FastF1. No se habilita como predictor base. Si más adelante se desea «mejor vuelta conocida hasta ahora», se derivará cronológicamente desde tiempos y decisiones oficiales disponibles antes de cada predicción, con un nombre distinto del campo retrospectivo.

## Decisión propuesta por columna

| Columna | Tratamiento del experimento base | Condición de calidad/tiempo |
|---|---|---|
| `LapTime` | Candidata RNN en vueltas completadas. | Cierre y disponibilidad ≤ decisión; máscara de ausente. |
| `Sector1Time` | Candidata RNN en vueltas completadas. | Auditar 858 posibles reconstrucciones algebraicas; no usar sector actual. |
| `Sector2Time` | Candidata RNN en vueltas completadas. | Mantener faltantes, cierre y disponibilidad temporal. |
| `Sector3Time` | Candidata RNN en vueltas completadas. | Mantener faltantes, cierre y disponibilidad temporal. |
| `Sector1SessionTime` | Alineación/auditoría, fuera de `X` crudo. | Marca de tiempo del sector, no de recepción; reloj validado. |
| `Sector2SessionTime` | Alineación/auditoría, fuera de `X` crudo. | Misma regla. |
| `Sector3SessionTime` | Alineación/auditoría, fuera de `X` crudo. | No exigir igualdad exacta con `Time`. |
| `SpeedI1` | Candidata RNN histórica. | Registrar posibilidad de *forward fill* y procedencia. |
| `SpeedI2` | Candidata RNN histórica. | Registrar posibilidad de *forward fill* y procedencia. |
| `SpeedFL` | Candidata RNN **solo histórica**; nunca token actual. | Faltante casi equivalente a `PitInTime` de la misma vuelta; controlar máscara y sensibilidad. |
| `SpeedST` | Candidata RNN histórica. | Punto de medición depende del circuito; revisar *forward fill*. |
| `IsPersonalBest` | Excluir del `X` base. | Derivar otro indicador as-of si se justifica. |
| `Position` | Excluir el valor consolidado del `X` base. | Reconstruir posición as-of con fuente temporal; no basta desplazar la fila. |

## Criterios para la transformación futura

1. Ningún tiempo, sector, velocidad, posición ni máscara estructural de la vuelta `N` calculado después del inicio entra en `X_N`.
2. La validación de `LapTime = S1 + S2 + S3` se ejecuta donde los cuatro valores existen; toda reconstrucción añade procedencia, revisión de casos extremos y versión nueva.
3. El tratamiento de `SpeedFL` se comprueba específicamente en vueltas `PitIn`: una prueba que permita la máscara actual debe fallar antes de publicar datos para entrenar.
4. Los faltantes de vueltas generadas y de sectores no se convierten en cero ni eliminan por defecto todas las observaciones del piloto.
5. La posición se obtiene de una fuente as-of verificable o se mantiene ausente en `X`; ninguna posición final o rango retrospectivo de la carrera cruza el límite temporal.

**Siguiente grupo propuesto:** estado de pista (`TrackStatus`, `SafetyCar`, `VirtualSafetyCar`) y meteorología. Sus valores de la fila actual son especialmente sensibles al tiempo de publicación.
