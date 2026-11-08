# Comparación de capacidad recurrente: RNN32 frente a RNN16

**Identificador de pareja:** `rnn16_vs_rnn32_f2_v1_seed42`.
**Estado:** pareja entrenada en Tesla T4 y evaluada localmente sobre 2024; ver [resultados](../aclaraciones/12_resultados_rnn16_pareada.md). La RNN16 no superó F2 de la referencia en esta semilla.
**Notebook:** [`notebooks/03_rnn16_colab.ipynb`](../../notebooks/03_rnn16_colab.ipynb).
**Referencia pareada:** [`configs/experimento_rnn32_pareado_v1.json`](../../configs/experimento_rnn32_pareado_v1.json), `experiment_id=rnn_reference_v1`.
**Candidata:** [`configs/experimento_rnn16_v1.json`](../../configs/experimento_rnn16_v1.json), `experiment_id=rnn_hidden16_f2_v1`.

## Pregunta y control de la comparación

La pregunta es si reducir de 32 a 16 las unidades de la celda recurrente cambia el ajuste y F2 de validación, manteniendo las otras decisiones del experimento F2. La hipótesis es que una capacidad menor podría reducir la brecha entre entrenamiento y validación; debe comprobarse y puede resultar falsa.

Se entrenan **dos modelos nuevos en la misma sesión**: una referencia RNN32 y una candidata RNN16. Esta referencia actual es el control principal. La corrida histórica `runs/rnn_v1_seed42`, asociada al código `cfcfb95`, se conserva como contexto. Si el entorno actual difiere del histórico, la comparación con esa corrida incluye diferencias de entorno y no permite atribuir todo cambio a capacidad.

El único factor material cambiado dentro de la pareja es `model.hidden_size: 32 -> 16`. Los campos de identidad, rol, acompañamiento y comparación documentan la pareja; no añaden entradas ni cambian el aprendizaje. La corrida F1 y los archivos originales de F2 permanecen como experimentos separados.

## Configuraciones y decisiones fijas

Ambas configuraciones parten de `configs/experimento_rnn_v1.json`. La referencia conserva hidden size 32 y la candidata usa 16; las dimensiones restantes y las secciones completas de entrenamiento, selección y evaluación son iguales al protocolo F2 original.

| Componente | Control común |
|---|---|
| Datos | `EDA/dataset/muestras_preprocesadas_v1_3` |
| SHA-256 del manifiesto | `08afa3c1f406f6bbb33c27a5e688b0d0fd46658ec8398a095f8981d74abdbf66` |
| Características y ventanas | `muestras_causales_v1_3`, W=8, orden publicado |
| Etiquetas y particiones | v3 y `muestras_causales_v1_3_etiquetas_v3_particiones_v1_1` |
| Train / validación | Carreras completas 2022–2023 / 2024 |
| Población | `supervised_eligible=true`, etiqueta conocida y elegibilidad acordada |
| Preprocesador | Congelado, ajustado previamente con 46.162 muestras elegibles de train |
| Semilla | 42, inicializada para cada entrenamiento |
| Optimizador / pérdida | Adam / `BCEWithLogitsLoss` sin ponderación |
| Learning rate / batch | 0,001 / 256 |
| Presupuesto / paciencia | Máximo 30 épocas / 6 épocas sin mejorar checkpoint |
| Gradiente | Recorte de norma a 1 |
| Métrica principal | F2 positiva de validación |
| Checkpoint | F2 a umbral fijo 0,5; desempate por menor BCE de validación |
| Umbral | Cuadrícula 0,05–0,95, paso 0,01; pesos congelados |
| Desempate de umbral | Mayor precision, luego mayor umbral |
| Prueba final | 2025, reservada |

El máximo y la paciencia son comunes. La época elegida y la cantidad final de épocas pueden diferir entre modelos como resultado de sus respectivas curvas; no se impone que ambos paren en la misma época.

El loader verifica hashes, claves, alineación, esquema y orden de entradas antes de entrenar. No se recalculan muestras ni transformaciones, no se ejecuta `fit` y no se transforma ni evalúa 2025. No se altera el horizonte ni la población de validación.

## Qué significa RNN16

RNN16 cambia únicamente el estado oculto recurrente. Se mantiene una sola `RNNCell` tanh y la definición compartida `f1_pitstop.rnn_model.SmallPitStopRNN`.

| Parte del modelo | Referencia RNN32 | Candidata RNN16 |
|---|---|---|
| Numéricas dinámicas | 22 valores + 22 bits de disponibilidad | Igual |
| Categorías dinámicas | Dos embeddings de dimensión 4 | Igual |
| Entrada de la celda | 52 dimensiones | 52 dimensiones |
| Estado recurrente | 32 unidades | 16 unidades |
| Contexto estático | Tres embeddings de 4 → Linear 16 → ReLU | Igual |
| Fusión | 32 + 16 → Linear 16 → ReLU → un logit | 16 + 16 → Linear 16 → ReLU → un logit |
| Parámetros con vocabularios publicados | 4.121 | 2.233 |

Los conteos de parámetros se verifican sobre los modelos instanciados y quedan en el resumen de la pareja. No representan una métrica predictiva. La reducción de la dimensión de entrada a la fusión se deriva del estado recurrente menor; la cabeza conserva 16 unidades.

Cada muestra comienza con estado oculto cero. Padding, slots inexistentes y filas fuente ausentes conservan el estado anterior; no se compactan huecos ni se arrastra estado entre muestras. Los 22 bits son la codificación explícita de las máscaras de observación, y los conteos/fracciones de cobertura de QA permanecen fuera de X. PAD=0 y OOV=3 tienen embedding fijo cero; ausencia=1 y `UNKNOWN`=2 son aprendibles. No crecen vocabularios con validación.

## Selección de pesos y decisión

F2 es el criterio de selección; la función diferenciable de entrenamiento continúa siendo BCE. Después de cada época se calculan métricas a umbral fijo 0,5, con los pesos al cierre de esa época. El checkpoint maximiza F2 de validación y desempata por menor BCE. La parada temprana observa ese mismo criterio.

Una vez seleccionado cada checkpoint, se congelan sus pesos y se maximiza F2 sobre los 91 umbrales `0,05, 0,06, …, 0,95`. El desempate es mayor precision y luego mayor umbral. Cada modelo puede elegir un umbral distinto bajo el mismo procedimiento. No se elige otra época después de la búsqueda.

Se conservan F2, F1, precision, recall, AP, BCE, matriz de confusión y cantidad de alertas, junto con época elegida y umbral. Las métricas de train y validación a 0,5 permiten comparar con una misma regla de alerta; las métricas al umbral seleccionado permiten estudiar las decisiones de cada modelo. Una alerta requiere probabilidad mayor o igual al umbral y los cocientes sin denominador se reportan como cero.

## Entorno común dentro de la pareja

El entorno histórico registró Python 3.13.16 y PyTorch 2.14.1+cu130. Colab puede ofrecer otras versiones. No se instalan versiones históricas de forma obligatoria: el control experimental es la igualdad del entorno **entre los dos entrenamientos actuales**, con versiones exactas registradas.

Se recomienda GPU T4 si Colab la ofrece. CPU también permite una comparación pareada si ambos modelos usan el mismo entorno y dispositivo. No cambiar el tipo de entorno ni instalar paquetes entre corridas. La semilla y las opciones de ejecución se reinicializan bajo el mismo protocolo para cada modelo.

El runner recoge snapshots antes de la referencia, entre referencia y candidata, y después de la candidata. Registra además el entorno de cada hijo: Python y dependencias, dispositivo, GPU/CUDA, semilla, determinismo, hilos y configuración CUBLAS. `pair_environment.json` conserva `snapshots`, `children` y `comparison_authorized`. Si los controles no se cumplen, la comparación no se autoriza. Una misma semilla no garantiza igualdad bit a bit entre plataformas o versiones ni permite estimar variabilidad entre semillas.

## Ejecución en Colab y exportación

Con `requirements-experiments.txt` instalado, desde la raíz del repositorio:

```bash
python scripts/empaquetar_experimento_rnn.py \
  --config configs/experimento_rnn16_v1.json \
  --output runs/rnn_colab_rnn16_v1.zip
```

La configuración candidata declara `companion_configs=['configs/experimento_rnn32_pareado_v1.json']`. El ZIP incluye ambas configuraciones, código común y los artefactos necesarios de train/validación. Guardar el SHA-256 que muestra el empaquetador.

Abrir `notebooks/03_rnn16_colab.ipynb` en Colab. Si el ZIP ya está en `/content/rnn_colab_rnn16_v1.zip`, se reutiliza; `files.upload()` es la alternativa cuando falta. Si tiene otro nombre, ajustar `PACKAGE_PATH`. La extracción usa `/content/f1-rnn16`, exige destino nuevo y protege rutas, enlaces simbólicos y duplicados.

Antes de instalar o importar código se verifica `colab_package_manifest.json`: versión `rnn_colab_inputs_v1`, `test_arrays_included=false`, configuración principal candidata, declaración y hash de la referencia acompañante, conjunto exacto de archivos y SHA-256 de cada archivo. Las rutas del manifiesto deben permanecer dentro del destino. Se instala el archivo común de requisitos sin forzar wheels CPU.

El notebook muestra versiones y GPU/CPU y llama al comando compartido:

```bash
python scripts/entrenar_comparacion_capacidad.py \
  --reference-config configs/experimento_rnn32_pareado_v1.json \
  --candidate-config configs/experimento_rnn16_v1.json \
  --output runs/rnn16_vs_rnn32_f2_v1_seed42
```

La salida es nueva e incluye `rnn32_reference/`, `rnn16_candidate/`, `pair_summary.json`, `pair_environment.json`, `pair_manifest.json` y `status.json`. Cada hijo conserva el bundle estándar del modelo: pesos, arquitectura, configuración, preprocesador, umbral, historial, entorno, métricas y predicciones de validación por `sample_id`.

El notebook muestra los controles del entorno, el resumen y ambos historiales. Descarga `runs/rnn16_vs_rnn32_f2_v1_seed42.zip`, generado por el runner y sin ZIPs hijos anidados. Drive es una copia opcional; no es requisito del flujo. El notebook no duplica modelo, entrenamiento ni evaluación.

## Evaluación local de ambos hijos

Extraer el ZIP padre en una carpeta nueva. Desde la raíz del repositorio, ajustar los `--bundle` si se eligió otro destino:

```bash
python scripts/evaluar_rnn.py \
  --bundle runs/rnn16_vs_rnn32_f2_v1_seed42/rnn32_reference \
  --data EDA/dataset/muestras_preprocesadas_v1_3 \
  --output eval/rnn16_vs_rnn32_f2_v1_seed42/rnn32_reference \
  --device cpu

python scripts/evaluar_rnn.py \
  --bundle runs/rnn16_vs_rnn32_f2_v1_seed42/rnn16_candidate \
  --data EDA/dataset/muestras_preprocesadas_v1_3 \
  --output eval/rnn16_vs_rnn32_f2_v1_seed42/rnn16_candidate \
  --device cpu
```

Los bundles apuntan a las carpetas con `weights.pt`, no al ZIP padre. Ambos evaluadores usan la misma validación 2024, sus respectivos pesos y umbrales congelados y los mismos artefactos preprocesados. Contrastan probabilidades Colab/CPU por `sample_id` con `parity_atol=1e-5` y `parity_rtol=1e-4`. Una discrepancia fuera de tolerancia debe investigarse antes de publicar métricas. Se conservan diferencias y cantidad de decisiones que cambian cerca del umbral, incluso dentro de la tolerancia.

La cobertura por evento se calcula solo localmente. Requiere `events.csv` de `etiquetado_pit_entry_v3_2022_2025` y `sample_index.csv.gz` de `muestras_causales_v1_3`, ya existentes en las rutas configuradas y verificados por hash. No se incluyen en Colab. Se informa cobertura con sus oportunidades elegibles y horizonte, separada de F2 por muestra.

## Interpretación posterior

Comparar primero la pareja actual y registrar la diferencia RNN16 menos RNN32, conservando también resultados negativos y fallos. Una brecha de train/validación menor no basta para declarar mejora si empeora el criterio principal. El resultado de la referencia histórica sirve para contexto y requiere declarar las diferencias de entorno que existan.

Este experimento no cambia pérdida, métrica, ponderación, learning rate, regularización, entradas ni ventana. No asegura que menos capacidad mejore F2. Validación se usa para selección y sus resultados son de desarrollo; una sola semilla no mide estabilidad. **2025 permanece reservada** para una evaluación final acordada con protocolo y modelo congelados.

Referencias: [experimento F2](11_experimento_rnn_inicial.md), [experimento F1](12_experimento_rnn_f1.md), [diseño ortogonal](05_diseno_ortogonal_experimentos.md) y [preprocesamiento](10_contrato_preprocesamiento.md).
