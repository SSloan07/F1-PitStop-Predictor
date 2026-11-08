# SPEC-DATA-002: construcción de muestras causales

**Versión:** 1.0 · **Estado:** contrato ejecutado en muestras_causales_v1_3; no autoriza entrenamiento ni ajuste de preprocesadores.

Las publicaciones anteriores se conservan. La salida final muestras_causales_v1_3 además corrige dos defectos de estado de neumático: timing_app_data usa stints desde cero y la matriz desde uno, por lo que la correspondencia explícita es timing stint + 1; además, una actualización tardía de un stint anterior no revierte el stint activo, que se obtiene del mayor índice observado disponible. La edad del neumático se toma solo de vueltas cerradas del stint de matriz correspondiente. El reporte distingue etiquetas sin publicación de etiquetas publicadas desconocidas.

La versión también mantiene las garantías previas: métricas completas de N indisponibles al inicio de N; linaje con carrera y piloto; padding técnico con número 0 y máscara; retrasos configurables separados para métricas de vuelta, posición y telemetría.

## 1. Unidad y objetivo

Una muestra identifica `(Temporada, Ronda, DriverNumber, prediction_lap)`. `decision_time` es `LapStartTime` de esa vuelta, en reloj de sesión. El objetivo publicado conserva el horizonte fijo de tres vueltas: `N`, `N+1` y `N+2`, independiente de la ventana `W`.

La matriz normalizada v1 aporta observaciones de 2022–2025. La etiqueta vigente `etiquetado_pit_entry_v3_2022_2025` cubre esas cuatro temporadas con la regla de pareja obligatoria. En la alineación con v1_3 se hace unión izquierda por clave completa con cardinalidad uno a uno, conservando tres estados: `published_known`, `published_unknown` y `not_published`. Las etiquetas, `PitInTime` y `PitOutTime` permanecen fuera de X. Las etiquetas que siguen incrustadas en `muestras_causales_v1_3/labels_aligned.csv.gz` son una publicación anterior y no deben consumirse.

## 2. Ramas de entrada permitidas

### Contexto estático FFNN

| Campo | Unidad/tipo | Fuente y disponibilidad |
|---|---|---|
| `Driver` | categoría cruda | matriz; contexto estático de la carrera |
| `Team` | categoría cruda | matriz; contexto estático de la carrera |
| `GranPremio` | categoría cruda | matriz; circuito/contexto de carrera |

No se aprende ni guarda vocabulario/codificación en esta etapa. `DriverNumber`, temporada y ronda son claves/procedencia, no valores de X.

### Secuencia RNN

Cada token conserva `lap_number` y dos grupos semánticos: estado que se conocía al inicio de esa vuelta y mediciones de esa vuelta, visibles solo una vez cerradas y disponibles.

| Grupo/campo | Unidad/tipo | Disponibilidad y regla |
|---|---|---|
| `Compound_asof` | categoría cruda, incluyendo `UNKNOWN` | `timing_app_data`, hasta `decision_time`, únicamente dentro del stint activo conocido. Una actualización parcial nula no borra el último valor conocido del mismo stint; un stint nuevo empieza desconocido. Nunca se hereda el stint anterior ni se usa `Compound` retrospectivo de la fila actual. |
| `TyreLife_last_eligible` | vueltas, numérico nullable | Último `TyreLife` observado de una vuelta completada y disponible del stint activo. No inferir de `FreshTyre`, `TotalLaps`, `StartLaps` ni vecinos futuros; preservar NA. |
| `track_phase_asof` | categoría de estado | Último `TrackStatus` con evento y disponibilidad no posteriores al inicio. Incluye `UNKNOWN`; los estados 4/5/6/7 se preservan como Safety Car, roja, VSC y VSC terminando. Estado 5 o sesión `Aborted` bloquea elegibilidad para alertar, pero se conserva la muestra. |
| Clima | °C, %, mbar, booleano, grados y m/s | Unión hacia atrás por carrera; `event_time <= decision_time`, `available_at <= decision_time` y edad `decision_time-event_time <= K_clima`. Base `K_clima=120 s`, configurable. Viento se representa como seno/coseno de grados. Una observación futura o expirada queda ausente con motivo/máscara. |
| `TyreLife`, `LapTime`, sectores, `Position`, `Telemetry_*` aprobados | según matriz; tiempo/sectores en segundos; posición ordinal | Solo fila cerrada con `Time + delay_de_fuente <= decision_time`, no generada y dentro de la misma secuencia piloto-carrera. Para la vuelta actual, las mediciones completas son temporalmente indisponibles. Cero físico conserva máscara de valor-presente; no se usa como padding. |

Candidatas numéricas habilitadas para materialización: `LapTime`, `Sector1Time/2Time/3Time`, `Position`, `Telemetry_Speed_mean/max`, `Telemetry_RPM_mean/max`, `Telemetry_Throttle_valid_mean/max`, `Telemetry_Brake_mean`, las variables meteorológicas de la tabla de registro y `TyreLife_last_eligible`. `LapNumber` conserva el progreso. `SpeedI1/I2/FL/ST`, `Telemetry_nGear_*`, DRS, `Telemetry_Samples`, conteos/fracciones de cobertura y cualquier columna no listada quedan fuera de X. Los conteos y coberturas pueden aparecer únicamente en auditoría.

`Stint_asof` y `FreshTyre_asof` siguen propuestos y se excluyen. Las máscaras son estructuras separadas de control/modelado, no columnas predictor automáticas.

## 3. Reloj, latencia y reinicios

El replay inicial declara latencia cero supuesta: `available_at = event_time`. La configuración permite un retraso fijo por fuente; el filtro temporal es siempre el mismo y exige evento y disponibilidad no posteriores a la decisión. Un tiempo de evento FastF1 no acredita latencia de recepción en vivo.

Se valida identidad de carrera mediante `session_info` y coincidencia con `GranPremio`. Estado de pista, clima, timing app y estado de sesión se leen solo de las cachés Race locales versionadas por hash. Durante los reinicios revisados de Australia 2023 y Mónaco 2024 se conserva cada número de vuelta. Si `Time` del cierre previo supera `decision_time`, sus mediciones quedan temporalmente enmascaradas; no se desplaza una vuelta antigua ni se corrigen timestamps. Estado rojo/sesión suspendida hace la muestra no elegible para alerta, sin borrar sus observaciones.

## 4. Ventanas y máscaras

`W` configurable cuenta la posición actual; el artefacto inicial usa `W=8` sin fijarlo como hiperparámetro de modelo. El horizonte de etiqueta sigue en tres vueltas. La ventana nominal es `N-W+1 … N`, con alineación a la derecha: el token actual siempre ocupa `W-1`. Las vueltas anteriores a la 1 son padding. Las identidades de vuelta se guardan literalmente; nunca se rellenan huecos con otra vuelta.

Se almacenan por separado:

- `padding_mask`: posición anterior a la primera vuelta.
- `slot_row_missing_mask`: la vuelta nominal existe cronológicamente, pero falta su observación en la fuente.
- `temporal_unavailable_mask`: el valor/fila existe pero su evento o disponibilidad cae después de la decisión, o la vuelta es generada/incompleta.
- `feature_missing_mask`: valor realmente ausente/inválido después de aplicar la regla de disponibilidad.
- `slot_exists_mask`: existe la posición de vuelta en el bloque temporal.

Los arreglos numéricos usan `0.0` solo como almacenamiento para valores enmascarados; la máscara distingue ese relleno técnico del cero físico. Categorías se almacenan como strings (`UNKNOWN` para estado desconocido, string vacío solo para padding/técnica). No se ajustan imputadores, escaladores, encoders ni umbrales.

## 5. Elegibilidad y conservación

No se eliminan muestras por etiqueta desconocida, falta de característica o generación. `supervised_eligible` refleja la etiqueta publicada y `EligibleForSupervisedLearning`; `prediction_eligible` refleja solo el estado conocido al decidir (por ejemplo, no roja/suspendida ni fila actual generada). Los motivos se guardan sin descartar el contexto histórico que otras ventanas pueden necesitar.

Claves incompletas/duplicadas, incompatibilidad de hashes/esquema o identidad de caché incongruente detienen publicación. Lagunas internas se representan en su posición nominal con máscara, sin desplazar la historia. Las columnas de etiquetas, pit events, banderas retrospectivas y procedencia no se copian a tensores X.

## 6. Entradas y productos

Entradas de la construcción histórica: matriz Compound normalizada v1, auditoría de calidad v2 y sus manifiestos, etiquetas `etiquetado_pit_entry_v2_pair_required`, cachés Race 2022–2025. Se validan SHA-256, esquemas y claves antes de escribir. Todas las fuentes son de solo lectura.

Por ausencia de `pyarrow` en el entorno de construcción, esta versión usa CSV UTF-8 comprimido para tablas y NPZ con arrays NumPy sin objetos/pickle para ventanas y máscaras. El paquete incluye `sample_index.csv.gz`, `labels_aligned.csv.gz`, `static_features.csv.gz`, `windows.npz`, `feature_registry.json`, `feature_sources.csv.gz`, `mask_codes.json`, configuración, informe de cobertura y manifiesto con hashes/versiones. Los arrays se alinean con `sample_index.sample_id`; se incluye ejemplo de carga sin `allow_pickle`.

La matriz de características y artefactos son candidatos de entrada reproducibles, no una matriz aprobada para entrenar. Particiones, vocabularios, imputación y escalado se ajustan después por experimento, solo con train.

## 7. Ejecución implementada

El comando ejecutable está en scripts/preparar_muestras_causales.py y la lógica en src/f1_pitstop/causal_samples.py. La configuración versionada está en configs/muestras_causales_v1.json. Ejecutar:

    python scripts/preparar_muestras_causales.py \
      --config configs/muestras_causales_v1.json \
      --output EDA/dataset/muestras_causales_v1_3

La carpeta de salida debe ser nueva. El comando comprueba hashes y versiones de entradas, identidad session_info de cada carrera, estructura y unicidad de claves. El resultado conserva todas las filas de la matriz normalizada 2022–2025 y en su publicación inicial unió las etiquetas entonces disponibles de 2022–2023. El manifiesto registra hashes de entradas, cachés consultadas y archivos resultantes. Las pruebas focalizadas están en scripts/test_muestras_causales.py.

## 8. Etiquetas y particiones vigentes

Las etiquetas actuales están en `EDA/dataset/datos_carreras_completos_2022_2025/etiquetado_pit_entry_v3_2022_2025/`. La invariancia histórica respecto de `etiquetado_pit_entry_v2_pair_required` fue comprobada para las 47.997 claves de 2022–2023. La alineación y las particiones vigentes están publicadas por separado en `EDA/dataset/muestras_causales_v1_3_etiquetas_v3_particiones_v1_1/`; el paquete referencia por hash `sample_index.csv.gz`, `static_features.csv.gz` y `windows.npz` de v1_3, sin volver a calcular ni copiar arrays. La publicación inicial `_v1` se conserva.

La partición cronológica vigente agrupa carreras completas: entrenamiento 2022–2023 (44 carreras), validación 2024 (24) y prueba final reservada 2025 (24). Se conservan todas las muestras en las tablas de partición. La selección supervisada requiere etiqueta conocida, `EligibleForSupervisedLearning` y `prediction_eligible`; faltantes de características y padding no excluyen muestras. Las etiquetas publicadas desconocidas y las ausencias de publicación tienen estados distintos. La prueba final no se usa para elegir variables, reglas, modelos ni umbrales.

Reproducción de alineación y particiones: `python scripts/preparar_particiones_causales.py --config configs/particion_pit_entry_v1.json`. El manifiesto registra SHA-256 de las etiquetas y artefactos de ventanas/contexto, listas exactas de `sample_id` por conjunto y versiones de software. Rehúsa sobrescribir una carpeta existente.

El preprocesamiento de entradas ya está especificado e implementado en [SPEC-DATA-003](10_contrato_preprocesamiento.md). Su estado ajustado y transformaciones de train/validación se publican en `muestras_preprocesadas_v1_3`, sin procesar valores de prueba. El siguiente paso es acordar el protocolo experimental y los detalles arquitectónicos antes de comparar el baseline MLP propuesto con la arquitectura recurrente. Esta especificación no autoriza entrenamiento ni evaluación predictiva.
