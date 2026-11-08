# Matriz de decisiones de calidad y preparación

**Versión:** 1.0 · **Fecha:** 2026-11-07
**Alcance:** comprensión y limpieza histórica de la matriz; no define entradas de modelos ni aprueba variables pendientes.
**Fuente perfilada:** `EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025.pkl` (maestra consolidada; no la candidata de compuesto reconstruido ni la salida limpieza v1).

Estados: **Acordado** = decisión vigente; **Propuesto** = opción para socialización; **Pendiente** = falta decisión/evidencia; **Descartado** = no se aplicará dentro del alcance actual. “Auditoría” indica que el dato se conserva y reporta, no que se convierta en regla de exclusión.

| Variable/campo | Fuente y significado | Disponibilidad conocida | Faltantes/anomalías: tratamiento actual | Estado y evidencia |
|---|---|---|---|---|
| `Temporada`, `Ronda`, `DriverNumber`, `LapNumber` | Matriz FastF1 consolidada; identifican carrera, piloto y vuelta observada. | Identidad del registro; no son características crudas. | Validar nulos y duplicados. Error explícito si hay problema; nunca escoger una fila ni eliminarla automáticamente. | **Acordado** como clave `(Temporada, Ronda, DriverNumber, LapNumber)`. Perfil actual: 101.290 filas, 0 claves nulas y 0 duplicadas. |
| `LapNumber` y continuidad | Índice de vuelta registrado por piloto-carrera. | No acredita que todas las vueltas estén presentes. | Reportar saltos y límites observados; no crear filas ni rellenar huecos. | **Acordado** como auditoría solamente; evidencia y excepciones en el reporte reproducible. |
| `PitInTime`, `PitOutTime` | Registros de entrada y salida del feed FastF1. Etiqueta de visita según SPEC-LABEL-002. | Históricos; no son predictores. `Time` de evento no es hora de recepción. | Preservar nulos y eventos incompletos. No imputar cero; pareja válida se exige para positiva; eventos huérfanos quedan inciertos. | **Acordado** en [SPEC-LABEL-002](../specs/07_objetivo_entrada_boxes.md). La revisión del código encuentra divergencias que se corrigen y prueban. |
| `Compound` | Compuesto reportado o heredado en la matriz. | La columna consolidada no prueba disponibilidad al inicio de la vuelta. | En una nueva matriz histórica derivada de la maestra, convertir exclusivamente los literales exactos `"None"` y `"nan"` a `pd.NA` con dtype nullable; registrar cada cambio en auditoría reversible. Conservar categorías explícitas (`UNKNOWN`, `TEST_UNKNOWN`) y compuestos físicos. Sin reconstrucción ni imputación. | **Acordado e implementado** como normalización histórica de representación, no como aprobación de matriz de entrenamiento. `Compound_asof` sigue siendo contrato separado de replay causal y usa `UNKNOWN` si no hay información elegible. Ver [aclaración de marcadores](../aclaraciones/03_evaluacion_marcadores_compound.md). |
| `TyreLife` | Edad del juego reportada por FastF1. | La fila actual no demuestra que el valor estuviera disponible al instante de predicción. | Mantener faltantes; no imputar desde `FreshTyre`, `TotalLaps` ni vecinos futuros. | **Acordado** `TyreLife_last_eligible` como candidata baseline; materialización causal y máscaras siguen pendientes. |
| `Stint` / `Stint_asof` | Índice de stint de matriz o contador reconstruible de eventos fuente. No equivale necesariamente a número de paradas. | Matriz consolidada es retrospectiva; `timing_app_data.Time` permite replay por tiempo de evento. | Registrar discrepancias y seis inicios de stint sin pareja de boxes; no inferir parada ni corregir contador. | `Stint_asof`: **Propuesto**, separado y no aprobado para baseline. `Time` no demuestra latencia real. |
| `FreshTyre` / `FreshTyre_asof` (`New`) | Indicador de juego nuevo en la matriz / feed de stint. | Estado as-of solo bajo replay histórico; fuente puede tener registros iniciales incompletos. | Preservar; valores no interpretables quedan desconocidos. No inferir de `TyreLife` ni copiar retrospectivamente el valor final. | `FreshTyre_asof`: **Propuesto** por separado para ablación posterior; no está aprobado ni incorporado al baseline. |
| `FastF1Generated` | Bandera de fila generada por FastF1. | Disponible en la matriz histórica; no constituye señal autónoma de abandono. | Conservar y auditar. No borrar filas ni etiquetar retiro automáticamente. | **Acordado** conservar; exclusión de filas como muestras requiere aplicar las reglas del objetivo y horizonte, no esta bandera sola. |
| `IsAccurate`, `Deleted`, `DeletedReason`, `IsPersonalBest` | Indicadores/metadatos de precisión o estado de vuelta. | No asumir que se conocen al inicio de la vuelta; algunos pueden ser retrospectivos. | Conservar para auditoría; no filtrar filas ni invalidar mediciones sin análisis específico. | **Acordado** conservar; disponibilidad predictiva y uso futuro quedan pendientes por campo. |
| `Telemetry_*` | Resúmenes por vuelta y cobertura de muestras. | Un resumen completo de vuelta no está disponible al inicio de esa misma vuelta. | Reportar faltantes, ceros y negativos sin reemplazarlos ni declarar inválidos; sin umbrales generales. | **Acordado** solo conservar y perfilar en limpieza; elegibilidad de predictores se trata en el contrato causal y lista candidata. |
| `Weather_*` | Observaciones meteorológicas enlazadas a la matriz. | La fila no prueba disponibilidad en tiempo de decisión. El tiempo de la observación se conserva como procedencia. | No imputar ni eliminar faltantes en esta etapa. La unión temporal debe excluir observaciones posteriores y considerar tolerancia configurada. | **Acordado** principio causal; el replay de FastF1 es latencia cero supuesta, no validación de vivo. |
| `SafetyCar`, `VirtualSafetyCar`, `TrackStatus` | Estado agregado asociado a la vuelta. | El valor por vuelta puede resumir eventos que ocurren después del inicio de esa vuelta. | Conservar en matriz maestra; no usar agregado retrospectivo como estado actual. | `track_phase_asof` es derivación acordada para modelado, pero separada de limpieza; requiere replay de Race Control. |
| `Position`, velocidades de timing | Resultado por vuelta/punto de medición, según campo. | La posición final de vuelta o resultado puede ocurrir después de decidir al inicio. | Conservar; no filtrar por faltante ni trasladar valores retrospectivos a la vuelta actual. | Historial de posición es candidata acordada; velocidades I1/I2/ST siguen pendientes según [lista candidata](19_lista_candidata_baseline.md). |
| Cualquier columna, dtype y unidad | Esquema observado del archivo fuente. | El esquema no es contrato semántico por sí solo. | Registrar nombres, tipos, nulos, cardinalidad y rangos numéricos; cambios de esquema fallan validación contra el contrato esperado. | **Acordado** para auditoría reproducible; la conversión de unidades o tipos semánticos queda pendiente si requiere interpretación. |

## Reglas descartadas en esta fase

- No se definirá la etiqueta como cambio de compuesto o aumento de `Stint`; el objetivo vigente es la visita emparejada de `PitInTime`/`PitOutTime`.
- No se eliminarán filas por `IsAccurate=False`, `FastF1Generated=True`, telemetría cero/extrema ni por pertenecer a Austria 2022 sin una regla aprobada por campo y evidencia específica.
- No se imputarán valores de manera global ni se reconstruirán predictores con observaciones posteriores al instante de decisión.

## Transformaciones aplicadas

La auditoría conservadora v1.3 permanece disponible y no transforma valores ni filas. Además, `scripts/normalizar_compound.py` materializa la decisión aprobada en una salida nueva de la matriz maestra, registra las filas transformadas, manifiesto y conciliación, y rehúsa sobrescribir. Esta salida histórica no agrega predictores ni etiquetas, y no sustituye la construcción causal de características.

La conversión es específica a la matriz consolidada. **No** se aplica como borrado de estado en `timing_app_data`: un `None` en una actualización parcial puede significar campo omitido. El replay conserva el estado conocido dentro del mismo stint y nunca lo hereda del stint anterior.

## Decisiones pendientes para una futura matriz procesada

1. Definir elegibilidad de filas generadas y horizontes incompletos sin usar el resultado final en predicciones previas.
2. Socializar por separado `Stint_asof` y `FreshTyre_asof`; conservar ambos como propuestas mientras no haya aprobación explícita.
3. Validar disponibilidad de cada familia, supuestos de retraso y máscaras para una reproducción causal. No presentar `Time` de FastF1 como hora de recepción real.
4. Aprobar tratamiento de anomalías específicas solo con evidencia de origen; ceros, valores extremos o banderas de precisión no bastan para borrado.

## Reproducción

Desde la raíz del repositorio, la auditoría conservadora sigue disponible con `python scripts/auditar_calidad_matriz.py` (o `python EDA/dataset/limpiar_matriz.py`); rehúsa sobrescribir la salida existente. La normalización aprobada se ejecuta con `python scripts/normalizar_compound.py --input RUTA_MAESTRA --output-dir NUEVA_CARPETA`, o mediante `python EDA/dataset/limpiar_matriz.py --normalize-compound --input RUTA_MAESTRA --output-dir NUEVA_CARPETA`. No reutilizar una carpeta existente.
