# Segundo experimento RNN: selección por F1 positiva

**Identificador:** `rnn_reference_f1_v1`.
**Carpeta de corrida:** `runs/rnn_f1_v1_seed42`.
**Estado:** corrida entrenada y evaluada localmente sobre 2024; resultados y comparación en [el informe 08](../aclaraciones/08_resultados_rnn_f1_y_comparacion.md). El entorno efectivo fue CPU y difiere de la referencia T4; el informe documenta esa limitación.
**Configuración:** [`configs/experimento_rnn_f1_v1.json`](../../configs/experimento_rnn_f1_v1.json).
**Notebook:** [`notebooks/02_rnn_f1_colab.ipynb`](../../notebooks/02_rnn_f1_colab.ipynb).
**Referencia F2:** `rnn_reference_v1`, corrida `runs/rnn_v1_seed42` y [configuración F2](../../configs/experimento_rnn_v1.json).

## Pregunta y cambio declarado

La pregunta es qué cambia al elegir checkpoint y umbral por **F1 de la clase positiva**, manteniendo los datos, arquitectura y entrenamiento de la referencia F2. El factor declarado en configuración es `selection_metric: F2 -> F1`. La corrida F2 y sus artefactos se conservan como referencia.

Para conteos TP, FP y FN:

\[
F_1 = \frac{2TP}{2TP + FP + FN}, \qquad
F_2 = \frac{5TP}{5TP + 4FN + FP}.
\]

F1 equilibra precision y recall; F2 da mayor peso a los falsos negativos. Cambiar el criterio no garantiza mejorar F1, precision, recall ni la utilidad de las alertas. Ambos puntajes se reportan sobre la misma población para interpretar el efecto observado.

**F1 es un criterio de selección.** La función diferenciable que aprende los pesos sigue siendo `BCEWithLogitsLoss` sin ponderación de clases. Se conserva Adam, learning rate y presupuesto. Elegir otra métrica puede cambiar la época seleccionada y el momento de parada temprana aunque la pérdida y los pasos de optimización sean iguales.

## Decisiones que permanecen fijas

El experimento copia la configuración F2 y cambia únicamente los metadatos de identidad/comparación y la métrica principal de selección. Las entradas se verifican por los mismos hashes, incluido el SHA-256 del manifiesto preprocesado `08afa3c1f406f6bbb33c27a5e688b0d0fd46658ec8398a095f8981d74abdbf66`.

| Componente | Valor conservado |
|---|---|
| Datos | `EDA/dataset/muestras_preprocesadas_v1_3` |
| Características y ventanas | `muestras_causales_v1_3`, W=8 y orden publicado |
| Etiquetas | `etiquetado_pit_entry_v3_2022_2025` |
| Particiones | `muestras_causales_v1_3_etiquetas_v3_particiones_v1_1` |
| Población | Muestras `supervised_eligible=true`, con etiqueta conocida y elegibilidad acordada |
| Train / validación | Carreras completas 2022–2023 / 2024 |
| Preprocesador | Congelado, ya ajustado con 46.162 muestras elegibles de train |
| RNN | Una `RNNCell` tanh, hidden size 32 |
| Embeddings | Dimensión 4 en las dos categorías dinámicas y tres estáticas |
| Rama estática | Concatenación de 12 → Linear 16 → ReLU |
| Fusión | Estado 32 + estática 16 → Linear 16 → ReLU → un logit |
| Semilla | 42 |
| Entrenamiento | Adam, learning rate 0,001, batch 256, máximo 30 épocas |
| Pérdida / gradiente | BCE sin ponderación / recorte de norma a 1 |
| Parada temprana | Paciencia 6, observando el criterio de checkpoint |
| Dispositivo | `auto`, CUDA disponible o CPU |
| Prueba final | 2025, reservada |

La arquitectura compartida sigue siendo `f1_pitstop.rnn_model.SmallPitStopRNN`. La entrada recurrente mantiene 22 valores numéricos, sus 22 bits explícitos de disponibilidad y dos embeddings de dimensión 4: 52 dimensiones. El estado empieza en cero por muestra y no se actualiza en padding, slots inexistentes o filas fuente ausentes. No se compactan huecos ni se arrastra estado entre muestras.

PAD=0 y OOV=3 mantienen embeddings fijos cero; ausencia=1 y `UNKNOWN`=2 son aprendibles. Los vocabularios no crecen con validación. Las máscaras conservan el significado publicado y los conteos/fracciones de cobertura de QA no entran en X. No se modifica el preprocesamiento ni se ejecuta un nuevo `fit`.

## Selección de checkpoint y de umbral

Después de cada época se calculan métricas de train y validación con umbral **fijo 0,5**, incluidos F1 y F2. Las métricas de train se recalculan con los pesos al cierre de la época, para que la comparación use una misma versión del modelo.

El checkpoint seleccionado maximiza **F1 positiva de validación a 0,5**. Si F1 empata, se conserva menor BCE de validación. La paciencia de seis épocas se reinicia cuando mejora ese mismo criterio. No se elige una época usando el umbral posteriormente ajustado.

Una vez seleccionado el checkpoint, sus pesos se congelan. Se evalúan los 91 umbrales `0,05, 0,06, …, 0,95` sobre sus probabilidades de validación y se maximiza F1. Los empates se resuelven por mayor precision y luego mayor umbral. Se emite alerta con probabilidad mayor o igual al umbral; los cocientes sin denominador se reportan como cero.

Se exportan métricas de train y validación a 0,5 y al umbral elegido: F1, F2, precision, recall, matriz de confusión y cantidad de alertas. La diferencia entre ambos umbrales, dentro de un mismo checkpoint, mide el efecto de la regla de decisión.

## Comparación con la referencia y con el control de umbral

El control anterior de F1 sobre el **checkpoint F2 congelado** cambia únicamente el umbral de ese modelo. Permite preguntar cuánto del cambio puede obtenerse moviendo la regla de alerta, sin volver a entrenar ni seleccionar otros pesos.

Ese control usa el checkpoint F2 de la época 10 y las probabilidades exportadas. Sus resultados de validación ya calculados son:

| Decisión sobre los mismos pesos F2 | Umbral | F1 | F2 | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Umbral elegido por F2 | 0,12 | 0,279706 | 0,444631 | 0,172850 | 0,732613 | 8.116 | 619 |
| Umbral elegido por F1 | 0,22 | 0,296401 | 0,422191 | 0,198053 | 0,588769 | 5.519 | 952 |

Estas filas describen un cambio de decisión en el modelo F2 existente. No son resultados de la nueva corrida `rnn_reference_f1_v1`. La evidencia y los conteos completos se conservan en la [nota del control](../aclaraciones/07_control_f1_sobre_pesos_f2.md) y su [JSON de evidencia](../aclaraciones/evidencia_control_f1_v1/control.json).

Esta segunda corrida vuelve a entrenar con el mismo protocolo y selecciona época/parada por F1. La comparación debe distinguir:

| Comparación | Qué permite observar |
|---|---|
| Checkpoint F2, umbral F2 vs. umbral F1 | Efecto de la decisión con pesos congelados |
| Checkpoint F2 vs. checkpoint F1, ambos a 0,5 | Diferencia entre los pesos elegidos por los criterios |
| Checkpoint F2 y checkpoint F1, ambos con búsqueda F1 acordada | Diferencia entre modelos bajo una regla común de selección de decisión |

Conservar F1 y F2 en todas las filas de comparación, junto con precision y recall. También conservar época elegida, épocas ejecutadas, umbral y entorno. No atribuir a la arquitectura un cambio producido por selección o umbral: la arquitectura es la misma.

El historial de la primera corrida también alcanza su mayor F1 a umbral 0,5 en la época 10. Repetir con selección F1 podría conservar los mismos pesos si se reproduce la trayectoria: cambiar el criterio no obliga a elegir otra época. Este hecho vuelve especialmente útil el control de umbral con pesos congelados.

Para atribuir diferencias al criterio, conservar también el entorno de la referencia cuando sea posible: Python 3.13.16, PyTorch 2.14.1+cu130, NumPy 2.4.6, pandas 2.2.3 y GPU T4. Si cambia el entorno, registrarlo como una diferencia adicional y limitar la comparación; la semilla por sí sola no lo controla.

Una sola semilla no mide variabilidad. La semilla fija tampoco garantiza igualdad bit a bit entre CPU/GPU o versiones de PyTorch. Los resultados de validación son resultados de desarrollo, con selección sobre el propio conjunto, y no constituyen una evaluación final independiente.

## Flujo local → Colab → CPU local

Desde la raíz del repositorio, en un entorno con `requirements-experiments.txt` instalado:

```bash
python scripts/empaquetar_experimento_rnn.py \
  --config configs/experimento_rnn_f1_v1.json \
  --output runs/rnn_colab_f1_v1.zip
```

Guardar el SHA-256 que muestra el empaquetador. Abrir el notebook F1 en Colab y, si está disponible, activar GPU desde **Entorno de ejecución → Cambiar tipo de entorno de ejecución**. El paquete se puede subir previamente desde el panel Archivos a `/content/rnn_colab_f1_v1.zip`; la primera celda reutiliza ese archivo y solo llama a `files.upload()` cuando falta. Si el archivo tiene otro nombre, ajustar `PACKAGE_PATH`.

La extracción usa `/content/f1-rnn-f1`, con protección de rutas y enlaces simbólicos, y exige un destino nuevo. Antes de instalar o importar código se valida `colab_package_manifest.json`: versión `rnn_colab_inputs_v1`, `test_arrays_included=false`, configuración F1 declarada, conjunto exacto de archivos, rutas dentro del destino y SHA-256 por archivo. El ZIP contiene solo los artefactos necesarios de train y validación.

La instalación conserva los rangos de `requirements-experiments.txt`, sin forzar wheels CPU. Se registran las versiones exactas utilizadas. Drive es opcional y no es requisito para entrenar o descargar.

El notebook importa el modelo común y ejecuta:

```bash
python scripts/entrenar_rnn.py \
  --config configs/experimento_rnn_f1_v1.json \
  --output runs/rnn_f1_v1_seed42
```

Se muestra `history.csv` y `metrics.json`, y se descarga `runs/rnn_f1_v1_seed42.zip`, generado por el entrenamiento. La salida debe ser nueva. El paquete conserva pesos, arquitectura, configuración F1, preprocesador, umbral, entorno, historial, predicciones de validación y manifiesto con hashes. Ninguna celda duplica el modelo o la lógica de entrenamiento/evaluación.

Después de descargar y extraer la corrida, volver a la raíz del repositorio:

```bash
python scripts/evaluar_rnn.py \
  --bundle RUTA_DIRECTORIO_MODELO_F1 \
  --data EDA/dataset/muestras_preprocesadas_v1_3 \
  --output eval/rnn_f1_v1_validation \
  --device cpu
```

`--bundle` apunta a la carpeta con `weights.pt`, no al ZIP. Se reutilizan el modelo, pesos, preprocesador y umbral exportados, y se contrastan probabilidades por `sample_id` con `parity_atol=1e-5` y `parity_rtol=1e-4`. Una discrepancia fuera de tolerancia debe investigarse antes de publicar métricas. Probabilidades próximas al umbral pueden cambiar de decisión dentro de la tolerancia; el evaluador conserva diferencias y cuenta esos cambios.

La cobertura por evento se calcula **solo en el evaluador local**. Necesita `events.csv` de la publicación `etiquetado_pit_entry_v3_2022_2025` y `sample_index.csv.gz` de `muestras_causales_v1_3`, ya existentes en las rutas fijadas en configuración y verificados por hash. Esas fuentes no se incluyen en Colab. Se usa únicamente la población de validación y se informa el horizonte y las oportunidades evaluables; la cobertura por evento no sustituye F1 o F2 por muestra.

## Alcance de la siguiente decisión

Este experimento no cambia capacidad, entradas, ventana, ponderación de clases, learning rate ni regularización. Esos cambios serían comparaciones posteriores con otra pregunta declarada. Conservar también intentos fallidos y resultados negativos. Ningún resultado de validación habilita por sí solo a abrir 2025 ni permite afirmar de antemano que F1 ofrecerá mejores alertas.

Referencias: [experimento F2](11_experimento_rnn_inicial.md), [diseño ortogonal](05_diseno_ortogonal_experimentos.md), [muestras causales](09_construccion_muestras_causales.md) y [preprocesamiento](10_contrato_preprocesamiento.md).
