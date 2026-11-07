# Revisión de cobertura y calidad de la matriz completa

**Estado:** diagnóstico inicial de la descarga completa; no representa una matriz limpia ni contiene etiquetas validadas. Fuente: `datos_carreras_completos_2022_2025/vueltas_2022_2025.pkl`, 101.290 filas y 61 columnas.

## Cobertura

| Temporada | Carreras | Filas | Filas `FastF1Generated` | `IsAccurate=False` |
|---|---:|---:|---:|---:|
| 2022 | 22 | 23.577 | 41 | 16,55 % |
| 2023 | 22 | 24.420 | 38 | 12,83 % |
| 2024 | 24 | 26.604 | 26 | 11,43 % |
| 2025 | 24 | 26.689 | 30 | 13,66 % |
| **Total** | **92** | **101.290** | **135** | **13,55 %** |

No hay nulos en las claves `Temporada`, `Ronda`, `DriverNumber`, `LapNumber`, ni duplicados de esas cuatro columnas. La temporada 2025 tiene 19–20 pilotos distintos por carrera; no se infiere falta de carrera a partir de ese conteo porque puede reflejar retiros o no participación.

## Valores faltantes y eventos

`PitInTime` y `PitOutTime` faltan en 96,58 % de filas, lo esperado al ser eventos esporádicos, no mediciones continuas. `SpeedFL` falta en 3,56 % y sus faltantes se deben analizar junto con vueltas de entrada y salida. `TyreLife` falta en 989 filas: 520 (2022), 35 (2023), 0 (2024) y 434 (2025). Hay 426 valores literales `None` en `Compound`; investigar su distribución antes de normalizarlos. No imputar ni eliminar globalmente.

## Calidad y telemetría

Hay 135 filas generadas por FastF1; no todas significan abandono. `IsAccurate=False` aparece en 13.725 filas e incluye eventos de boxes, neutralizaciones y otras vueltas que no conviene descartar sin analizar el propósito.

Cuarenta filas no tienen muestras válidas de acelerador tras excluir 104: MAG Miami 2022 (1), BOT Japón 2022 (26), HAD Australia 2025 (1), RUS Baréin 2025 (12). Sus promedios válidos quedan faltantes; conservar origen y banderas de calidad.

La matriz no contiene `PitEnProximas3Vueltas`, de acuerdo con la decisión de no usar la entrada a boxes como etiqueta definitiva.

## Alineación temporal

Hay clima asignado en todas las filas, pero `Weather_Time` es posterior a `LapStartTime` en 99,97 % de filas. Aplicar `K_clima` y un as-of temporal por instante de alerta; la presencia del clima asociado a la vuelta no implica que estuviera disponible al iniciarla. Los resúmenes de telemetría describen la vuelta completa y tampoco pueden usarse para predecir al inicio de esa misma vuelta.

## Pendientes antes de limpiar o entrenar

1. Auditar calendarios, carreras por ronda y pilotos; entender casos de 19 pilotos y filas generadas.
2. Revisar los 434 faltantes de `TyreLife` en 2025 y los 426 `Compound='None'` por carrera/stint; no extender la decisión sobre Austria sin análisis actualizado.
3. Revisar 13 filas sin acelerador válido en 2025, en especial RUS Baréin, contra telemetría original.
4. Diseñar alineación as-of de clima con `K_clima`; mantener telemetría de vuelta completa solo para predicciones posteriores.
5. Validar etiquetado de cambio de neumáticos con fuentes independientes y definir cobertura del horizonte.
6. Generar reporte reproducible por carrera y temporada antes de publicar la matriz procesada.

Este diagnóstico es agregado y no comprueba todavía integridad causal/temporal por fila. La matriz completa es materia prima de investigación, no el dataset listo para entrenamiento.
