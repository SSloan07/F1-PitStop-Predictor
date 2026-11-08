# Primer experimento RNN: referencia reproducible en Colab

**Identificador:** `rnn_v1_seed42`.
**Estado:** protocolo e implementación inicial; entrenamiento y resultados pendientes. Este documento no presenta métricas experimentales.
**Configuración ejecutable:** [`configs/experimento_rnn_v1.json`](../../configs/experimento_rnn_v1.json).
**Notebook:** [`notebooks/01_rnn_inicial_colab.ipynb`](../../notebooks/01_rnn_inicial_colab.ipynb).

## Pregunta y criterio principal

La pregunta inicial es si una RNN sencilla, con una rama estática mínima, puede aprender a anticipar una entrada a boxes con el historial causal publicado. Esta corrida establece la referencia para comparaciones posteriores; no exige entrenar otro modelo previamente.

El objetivo principal es maximizar **F2 de la clase positiva** en validación, sobre muestras piloto-vuelta supervisadas elegibles. Para conteos TP, FP y FN:

\[
F_2 = \frac{5TP}{5TP + 4FN + FP}.
\]

F2 da más peso a las entradas omitidas que a las falsas alertas. El entrenamiento exporta métricas por muestra piloto-vuelta: precision, recall, F1, F2, matriz de confusión y cantidad de alertas. La cobertura por evento se calcula únicamente en el evaluador local; responde a otra pregunta y se presenta separada de F2 por muestra. Una probabilidad mayor o igual al umbral produce una alerta. Los cocientes sin denominador se reportan como cero.

## Datos y límites de esta corrida

Se consume exclusivamente el paquete congelado `EDA/dataset/muestras_preprocesadas_v1_3`, incluidos sus manifiestos, `preprocessor.json`, arrays, etiquetas y metadatos de train y validación. El loader verifica integridad y alineación por `sample_id`, esquema, dimensiones y orden antes del entrenamiento.

| Componente | Decisión fija |
|---|---|
| Características y ventanas | `muestras_causales_v1_3`, W=8, orden publicado |
| Etiquetas | `etiquetado_pit_entry_v3_2022_2025` |
| Particiones | `muestras_causales_v1_3_etiquetas_v3_particiones_v1_1` |
| Train | Carreras completas de 2022–2023 |
| Validación | Carreras completas de 2024 |
| Prueba final | 2025, reservada |
| Selección supervisada | `supervised_eligible=true`, con etiqueta conocida y elegibilidad acordada |
| Preprocesador | Estado ya ajustado con 46.162 muestras elegibles de train |

El notebook no ejecuta `fit`, no reconstruye muestras y no transforma ni evalúa 2025. El paquete de Colab contiene los artefactos de train y validación necesarios para esta corrida. Las etiquetas históricas incrustadas en `muestras_causales_v1_3/labels_aligned.csv.gz` no son una entrada del experimento. Se preservan las máscaras y el cero técnico del preprocesamiento; no hay nueva imputación, escalado ni ampliación de vocabularios.

## Arquitectura y consumo explícito de máscaras

La definición compartida es `f1_pitstop.rnn_model.SmallPitStopRNN`. El notebook y los comandos de entrenamiento y evaluación usan ese mismo código.

| Rama | Representación |
|---|---|
| Numérica dinámica | 22 valores estandarizados y 22 bits de disponibilidad, uno por variable |
| Categórica dinámica | `Compound_asof` y `track_phase_asof`, embedding de dimensión 4 por campo |
| Recurrente | Una `RNNCell` con tanh, entrada de 52 dimensiones y estado oculto de 32 |
| Contexto estático | `Driver`, `Team`, `GranPremio`, embedding de dimensión 4 por campo; concatenación de 12 → `Linear(12,16)` → ReLU |
| Fusión | Estado recurrente de 32 + rama estática de 16 → `Linear(48,16)` → ReLU → `Linear(16,1)` |
| Salida | Un logit; sigmoid solo para probabilidades y decisiones |

Los 22 bits son una **codificación explícita de las máscaras de observación**, decisión de representación de esta arquitectura. Un bit vale uno solo para una medición numérica presente y disponible en una posición válida; vale cero para relleno técnico o ausencia. Esto distingue un cero físico estandarizado de un valor enmascarado. Los códigos de motivo y las máscaras originales siguen disponibles para validación y auditoría. Los conteos y fracciones de cobertura de QA no entran en X.

Cada muestra empieza con estado oculto cero. La celda recorre la ventana en orden cronológico y conserva el estado previo en padding, slots inexistentes o posiciones con fila fuente ausente. No compacta huecos ni arrastra estado entre muestras, pilotos o carreras. En una posición válida con medidas faltantes, recibe los ceros técnicos y sus bits de disponibilidad, junto con las categorías disponibles.

Los vocabularios y cardinalidades proceden de `preprocessor.json`: PAD=0 y OOV=3 tienen embedding fijo cero; ausencia=1 y `UNKNOWN`=2 tienen embeddings aprendibles. Las categorías conocidas conservan los IDs publicados. El OOV de validación no amplía el vocabulario ni recibe un vector aleatorio sin entrenamiento. Las representaciones estáticas y dinámicas usan esta misma regla.

## Entrenamiento y dos decisiones separadas

| Parámetro | Referencia inicial |
|---|---|
| Semilla | 42 |
| Presupuesto | Máximo 30 épocas |
| Batch size | 256 |
| Optimizador | Adam, learning rate 0,001 |
| Pérdida | `BCEWithLogitsLoss`, sin ponderación de clases inicial |
| Gradiente | Recorte de norma a 1 |
| Dispositivo | `auto`: CUDA disponible; CPU en su ausencia |
| Parada temprana | Paciencia de 6 épocas sin mejorar el criterio de checkpoint |

**Selección de pesos:** después de cada época se calcula F2 de validación con el umbral **fijo 0,5**. Se conserva el checkpoint con mayor F2; un empate se resuelve con menor pérdida de validación. La parada temprana observa exactamente este criterio. El entrenamiento no modifica la pérdida para intentar optimizar F2 directamente.

**Selección de decisión:** una vez elegido el checkpoint, se congelan los pesos y se evalúan los 91 umbrales `0,05, 0,06, …, 0,95` sobre sus probabilidades de validación. Se selecciona mayor F2; los empates se resuelven por mayor precision y luego mayor umbral. Esta búsqueda no cambia los pesos, no selecciona otra época y no usa 2025.

El historial registra pérdida, F2, precision y recall de entrenamiento y validación a umbral 0,5; las métricas de entrenamiento se recalculan con los pesos al cierre de cada época. Esto permite estudiar ajuste y generalización manteniendo la misma regla de alerta. La exportación conserva tanto las métricas con umbral 0,5 como las del umbral seleccionado sobre las mismas probabilidades, para train y validación. La diferencia entre ambas mide el efecto de la regla de alerta para ese checkpoint. El resultado ajustado en validación es una estimación de desarrollo; la evaluación final permanece pendiente y no se interpreta como una medición independiente de generalización.

## Flujo local → Colab → evaluación local

Los comandos locales requieren un entorno con `requirements-experiments.txt` instalado (`python -m pip install -r requirements-experiments.txt`). En la verificación local se usó Python 3.13 con PyTorch disponible; el Python 3.14 del sistema no tiene ese framework instalado.

Desde la raíz del repositorio, antes de abrir Colab:

```bash
python scripts/empaquetar_experimento_rnn.py \
  --config configs/experimento_rnn_v1.json \
  --output /tmp/rnn_colab_v1.zip
```

Abrir el notebook en Colab, escoger GPU si está disponible y ejecutar las celdas en orden. Una sesión CPU también es válida, con mayor tiempo de entrenamiento. El notebook solicita ese ZIP mediante `google.colab.files.upload`, muestra su SHA-256, verifica que las rutas del ZIP permanezcan dentro del destino y extrae en `/content/f1-rnn`. Después verifica `colab_package_manifest.json`: versión `rnn_colab_inputs_v1`, declaración `test_arrays_included=false`, rutas de archivos dentro del destino, conjunto exacto de archivos y SHA-256 de cada archivo. Estas comprobaciones ocurren antes de instalar dependencias o importar el código extraído. Si se conserva el hash local, puede comprobarse antes de extraer. No se necesita montar Google Drive.

El notebook instala `requirements-experiments.txt` sin forzar wheels CPU ni sustituir deliberadamente una instalación CUDA compatible. Los rangos permiten ejecutar en Colab; las versiones exactas de la corrida se registran en los artefactos exportados. Una semilla fija no implica igualdad bit a bit entre versiones, plataformas, CPU y GPU.

El comando de entrenamiento compartido es:

```bash
python scripts/entrenar_rnn.py \
  --config configs/experimento_rnn_v1.json \
  --output runs/rnn_v1_seed42
```

El notebook muestra el historial y las métricas, y descarga `runs/rnn_v1_seed42.zip`, generado por el comando de entrenamiento. La carpeta de corrida debe ser nueva para conservar la identidad de cada intento. Una copia opcional del ZIP en Drive permite conservarla cuando termine la sesión; la descarga directa es suficiente para continuar localmente.

Después de extraer el ZIP de la corrida, volver a la raíz del repositorio y ejecutar:

```bash
python scripts/evaluar_rnn.py \
  --bundle RUTA_DIRECTORIO_MODELO \
  --data EDA/dataset/muestras_preprocesadas_v1_3 \
  --output eval/rnn_v1_validation \
  --device cpu
```

El evaluador necesita además dos fuentes ya existentes en el repositorio, en las rutas fijadas en la configuración: `events.csv` de `etiquetado_pit_entry_v3_2022_2025` y `sample_index.csv.gz` de `muestras_causales_v1_3`. Se verifican por hash y no se incluyen en el paquete de Colab. Proporcionan los eventos y tiempos de decisión necesarios para calcular cobertura por evento de validación únicamente en la evaluación local.

`RUTA_DIRECTORIO_MODELO` es la carpeta que contiene `weights.pt`, no el ZIP. La evaluación local carga pesos, arquitectura, vocabularios, preprocesador y umbral congelados. Reproduce las probabilidades de validación y las contrasta por `sample_id` con las exportadas, con `parity_atol=1e-5` y `parity_rtol=1e-4`, declaradas en configuración y exportación. Si no satisface esa comparación, la evaluación falla antes de publicar métricas. Una diferencia superior a la tolerancia debe investigarse; no se reajusta el modelo ni el umbral para ocultarla. Probabilidades muy próximas al umbral pueden cambiar de lado incluso dentro de la tolerancia: el evaluador conserva las diferencias y la cantidad de decisiones que cambian.

## Artefactos y evidencia

La corrida exporta `weights.pt`, `model_spec.json`, `threshold.json`, `preprocessor.json`, configuración, entorno exacto, `history.csv`, `metrics.json`, `validation_predictions.csv.gz` y un `manifest.json` con hashes. Se registra la época elegida y su criterio, las versiones de datos, orden de entradas, semillas y predicciones identificables por muestra. El manifiesto permite comprobar que la evaluación local usa los mismos parámetros y artefactos. Los archivos de métricas y predicciones constituyen la evidencia; este documento no anticipa sus valores.

Las métricas auxiliares y el análisis de errores por carrera y piloto permiten interpretar FP y FN. El `metrics.json` del entrenamiento contiene métricas por muestra; la cobertura por evento se incorpora solo en la salida de evaluación local, con su población elegible y horizonte. No se infiere del promedio por muestra ni se usa como sustituto de F2.

## Ortogonalización para las siguientes iteraciones

La configuración inicial queda como referencia. Una variación posterior declara su pregunta y el factor cambiado: capacidad recurrente, representación de entradas, entrenamiento o decisión. Conserva datos, particiones, preprocesador, orden de entradas, W, métrica, semilla y presupuesto cuando no sean el factor estudiado. Los cambios conjuntos requieren declarar la interacción que se pretende medir.

Se conservan intentos fallidos y resultados negativos. Comparar pesos exige el mismo protocolo de selección; comparar umbrales exige pesos congelados. Esta primera corrida usa una sola semilla y no mide variabilidad entre semillas. Ninguna observación de validación habilita por sí sola a abrir 2025.

Referencias locales: [aclaración del primer experimento](../aclaraciones/05_primer_experimento_rnn_f2_colab.md), [diseño ortogonal](05_diseno_ortogonal_experimentos.md), [muestras causales](09_construccion_muestras_causales.md) y [preprocesamiento](10_contrato_preprocesamiento.md).
