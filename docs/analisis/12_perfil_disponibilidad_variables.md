# Perfil de variables y disponibilidad al predecir

**Matriz inspeccionada:** `vueltas_2022_2025_limpieza_v1.pkl`, temporadas 2022–2023.  
**Etiquetas asociadas:** `etiquetado_pit_entry_v19/labels.pkl`.  
**Estado:** análisis de disponibilidad y calidad; no es todavía la matriz de entrenamiento ni aprueba automáticamente predictores.

## Alcance y tamaños

La matriz tiene 47.997 filas y 61 columnas para 44 carreras. Las etiquetas tienen 47.997 filas: 4.990 positivas, 41.361 negativas y 1.646 desconocidas. Entre las 46.351 etiquetas conocidas, el 10,77 % es positivo (2022: 10,58 %; 2023: 10,95 %). Esta prevalencia es descriptiva; no define una métrica ni un umbral de alerta.

Los 1.646 `NA` se explican por cobertura del extremo de cada secuencia: 1.567 tienen horizonte incompleto (732 en la última vuelta observada, 779 a una vuelta y 56 a dos vueltas del final); otras 79 son muestras inválidas en la última vuelta. No aparecieron desconocidas a más de dos vueltas del final observado. Se mantienen como nulas, nunca como negativos.

## Inventario funcional de las 61 columnas

| Grupo | Columnas | Tratamiento inicial para modelado |
|---|---|---|
| Identidad/contexto | `Temporada`, `Ronda`, `GranPremio`, `Driver`, `DriverNumber`, `Team` | Contexto estático dentro de la carrera. Hay redundancia entre nombres, número y evento. Tratar como categorías/identificadores, nunca como magnitudes ordinales. Evaluar en experimentos estáticos; pueden causar memorización y no representan propiedades físicas del circuito. |
| Índice y reloj | `LapNumber`, `LapStartTime`, `LapStartDate`, `Time` | Derivar progreso de carrera y tiempo transcurrido usando solo el instante conocido. `LapStartTime` es el ancla propuesta de decisión; `Time` marca el tiempo de la fila al cierre de vuelta y no debe alimentar la predicción de inicio de esa vuelta. No usar fechas absolutas sin justificación. |
| Estado de neumático | `Stint`, `Compound`, `TyreLife`, `FreshTyre` | Candidatos dinámicos valiosos. En cada decisión, recuperar el último estado observable antes del instante de alerta; no asumir que el valor de la fila actual ya era conocido al inicio. Conservar nulos y edad del dato. |
| Entrada/salida de boxes | `PitInTime`, `PitOutTime` | Solo para construir/auditar eventos y etiquetas. Excluir de todas las entradas del modelo por fuga directa. |
| Tiempos y sectores | `LapTime`, `Sector1Time`, `Sector2Time`, `Sector3Time`, `Sector1SessionTime`, `Sector2SessionTime`, `Sector3SessionTime` | Resumen de vueltas ya transcurridas. Para la predicción al inicio de N, solo usar datos de vueltas cerradas (máximo N−1), con faltantes explícitos. Los tiempos de sectores actuales aparecen a medida que se completan y requerirían otro instante de predicción definido. |
| Velocidades de timing | `SpeedI1`, `SpeedI2`, `SpeedFL`, `SpeedST` | Máximos/puntos de medición asociados a la vuelta y potencialmente conocidos solo durante/después de ella. Usar con rezago o reconstruir snapshots causales; no usar los de la vuelta que empieza. |
| Telemetría agregada | `Telemetry_Samples`, `Telemetry_Speed_mean/max`, `Telemetry_RPM_mean/max`, `Telemetry_Throttle_mean/max`, `Telemetry_Brake_mean/max`, `Telemetry_nGear_mean/max`, `Telemetry_DRS_mean/max`, `Telemetry_Throttle_valid_samples/fraction/mean/max` | Agregados calculados sobre la telemetría de vuelta completa por `consolidar_carreras.py`; usar únicamente para vueltas cerradas. Aún requieren reglas de rango, calidad y codificación antes de aprobarlos. |
| Estado de pista | `TrackStatus`, `SafetyCar`, `VirtualSafetyCar` | El script construye las dos banderas buscando códigos en todo el `TrackStatus` de la vuelta. Es una agregación retrospectiva: no equivale al estado conocido al inicio. Para tiempo real, reconstruir el estado con Race Control hasta el instante de alerta; como alternativa histórica sencilla, usar el estado de la vuelta anterior. `TrackStatus` es código categórico, no número continuo. |
| Meteorología | `Weather_Time`, `Weather_AirTemp`, `Weather_Humidity`, `Weather_Pressure`, `Weather_Rainfall`, `Weather_TrackTemp`, `Weather_WindDirection`, `Weather_WindSpeed` | La selección actual de FastF1 asocia una observación por vuelta, pero no necesariamente disponible al inicio. Alinear con el último registro de clima con timestamp ≤ decisión y aplicar `K_clima`; guardar edad de la observación. Dirección de viento es circular (0°≈360°), no lineal. |
| Posición y resultado posterior | `Position`, `IsPersonalBest`, `Deleted`, `DeletedReason` | Excluir el valor de la vuelta actual: posición puede representar estado posterior del timing y mejor vuelta/eliminación son conocidos al completar o revisar la vuelta. `Deleted`/`DeletedReason` reflejan decisiones posteriores. Usar históricos rezagados solo si su semántica temporal se valida. |
| Calidad/procedencia | `FastF1Generated`, `IsAccurate` | Conservar para elegibilidad, auditoría y análisis de robustez. No son variables de rendimiento; pueden revelar el cierre o la calidad retrospectiva de la vuelta. No eliminar filas automáticamente. |

La etiqueta `PitWithin3Laps` vive en `labels.pkl`, no en la matriz maestra de 61 columnas. En el entrenamiento deberá unirse por `(Temporada, Ronda, DriverNumber, LapNumber)` y seleccionarse como `y`, nunca como `X`.

## Faltantes observados

| Columna(s) | Faltantes | Lectura |
|---|---:|---|
| `PitInTime`, `PitOutTime` | 46.218 (96,29 %), 46.233 (96,32 %) | Eventos esporádicos; no imputar ni contar como nulos de predictor. Son columnas de auditoría excluidas de `X`. |
| `Sector1Time`, `Sector1SessionTime` | 1.041 (2,17 %), 1.147 (2,39 %) | Faltantes de sectores; conservar; solo son causales tras completar el sector/vuelta. |
| `SpeedFL` | 1.857 (3,87 %) | Ausencia de speed trap/observación; no reemplazar por cero. |
| `Sector2Time`, `Sector3Time`, `Sector2SessionTime`, `Sector3SessionTime` | 89 (0,19 %), 178 (0,37 %), 89, 178 | Conservar nulos, reportar por carrera y no completar con estadísticas globales. |
| `LapTime`, `Position` | 270 (0,56 %), 79 (0,16 %) | Incluye vueltas no completadas/generadas; no significa que el auto estuviera quieto durante la vuelta. |
| `TyreLife`, `Compound` | 555 (1,16 %), 36 (0,08 %) | Estado de neumático incompleto; no imputar hasta validar por stint y disponibilidad temporal. |
| `SpeedI1`, `SpeedI2`, `SpeedST` | 80, 79, 113 | Faltantes raros, conservar bandera/NA. |
| `IsPersonalBest`, `DeletedReason` | 64 cada una | Campos retrospectivos; se excluyen como predictores actuales. |
| `Telemetry_Throttle_valid_mean/max` | 27 cada una | No hubo muestra válida en esas vueltas; el faltante es informativo. |

Las demás columnas no tienen nulos pandas en este corte. Esto no significa que tengan calidad suficiente ni que su valor estuviera disponible en la decisión.

## Calidad de telemetría: hallazgos para investigar

- `Telemetry_Throttle_mean` contiene 29 valores mayores a 100 y `Telemetry_Throttle_max` tiene 427 valores `104`; FastF1 define 104 como acelerador no disponible en el propio extractor. El conjunto `_valid_*` filtra valores fuera de 0–100; para experimentos usar esas columnas validadas o recomputar desde fuente, no los agregados sin filtrar.
- `Telemetry_nGear_max` supera 8 en 29 filas y `Telemetry_nGear_mean` supera 8 en 8; valores extremos observados llegan a 128 y 39,75. Son incompatibles con una marcha física de F1. Marcar/recalcular desde telemetría original antes de usar; no recortar silenciosamente.
- `Telemetry_DRS_max` usa códigos discretos (0, 1, 2, 3, 8–15) y no una escala continua; por ello `DRS_mean` tampoco tiene interpretación inmediata de magnitud. Validar el diccionario de códigos y crear variables derivadas con semántica explícita.
- Hay 80 vueltas con más de 1.000 muestras (25 con más de 5.000), 27 con velocidad media exactamente cero y 28 con RPM medias cero. Revisar trazas y cobertura antes de decidir si son observaciones válidas, vueltas de retiro o problemas de agregación.
- Hay 651 vueltas con `Deleted=True`, 7.037 con `IsAccurate=False` y 79 `FastF1Generated=True`. No son criterios de descarte automático: conservar banderas y estudiar su efecto por separado.
- El compuesto aparece como HARD (20.170), MEDIUM (17.540), SOFT (7.657), INTERMEDIATE (2.170), WET (424) y ausente (36). `TyreLife` falta en 555 filas; su distribución por compuesto no justifica imputación simple.

## Comprobación temporal de meteorología

Se calculó `Weather_Time − LapStartTime` en las 47.997 filas. La observación asignada es posterior al inicio de vuelta en 47.973 filas (99,95 %), igual en 1 y anterior en 23. La diferencia mediana es 30 segundos; percentiles 90 y 99: 53,92 s y 59,42 s. Además, el timestamp de clima cae durante el intervalo de la vuelta en 47.973 filas.

Por tanto, las columnas climáticas actuales de la fila no son predictores válidos al inicio de esa misma vuelta. Crear una tabla de observaciones de clima por carrera y hacer una unión `as-of` hacia atrás. En la salida, almacenar `Weather_Time`, edad en segundos y un indicador de expiración según `K_clima`. El valor cero de faltantes en las ocho columnas climáticas es consecuencia de asignar observaciones; no prueba disponibilidad contemporánea.

## Disponibilidad propuesta por experimento

1. **Rama estática FFNN (candidata):** identidad/contexto categórico conocido al inicio (carrera, piloto y equipo). Son identificadores de contexto, no características físicas; comparar un modelo sin identidad y particiones agrupadas por carrera/temporada para medir memorización. En esta fuente no hay metadatos físicos del circuito como longitud o número de curvas.
2. **Rama dinámica RNN (candidata):** historial de vueltas ya completadas hasta N−1: telemetría validada, tiempos y estado de neumático observado, más el estado de pista/clima alineado al instante. La observación de N solo puede contener campos de estado conocidos al cruzar la línea; no los resúmenes de la vuelta N.
3. **Exclusión estricta:** `PitInTime`, `PitOutTime`, `PitWithin3Laps`, señales derivadas del evento futuro, `DeletedReason`, valores de la vuelta actual calculados después del cruce y estado final del piloto.
4. **Variables pendientes de semántica causal:** `Stint`, `Compound`, `TyreLife`, `FreshTyre`, `Position`, `TrackStatus`, SC/VSC y cualquier `IsAccurate`/`FastF1Generated`. No aprobar hasta construir o validar su estado “as of” el instante de alerta.

## Siguiente transformación

No limpiar ni imputar por columna de forma global todavía. El próximo entregable debe construir una vista de características con clave de muestra y etiqueta unidas, ventanas causales, alineación `as-of` de clima/estado de carrera y columnas explícitamente permitidas. Mantener la matriz maestra intacta y añadir un reporte de decisiones, exclusiones y faltantes. Antes de exportarla para Colab/Kaggle, revisar algunos pilotos/vueltas seleccionados y verificar que la ventana que ve el modelo termina en información realmente disponible al cruce de inicio.
