# Experimento LSTM causal frente a RNN32

**Estado:** corrida pareada ejecutada y reproducida localmente. LSTM no superó F2 en la semilla 42; consulta el [informe](16_resultados_lstm_rnn32.md). La prueba 2025 permanece reservada.

## Decisión recomendada

Sí conviene hacer la comparación como siguiente experimento de arquitectura, pero como hipótesis y no bajo el supuesto de que LSTM sea automáticamente más robusta. Propongo una LSTM unidireccional de una capa con 32 unidades, usando la misma rama FFNN estática y la misma cabeza de fusión que el modelo actual. El cambio declarado sería el tipo de celda recurrente, RNN tanh → LSTM.

No propongo una Bi-LSTM ni una red profunda para esta primera comparación. La predicción se hace al cierre de una vuelta y solo puede consumir historia disponible hasta ese punto. La rama dinámica debe recorrer la ventana en orden causal y preservar exactamente las máscaras: padding, slot ausente y fila fuente ausente no actualizan ni el estado oculto ni el estado de memoria LSTM. No se deben compactar huecos ni transportar estado entre muestras.

## Qué aportan los papers locales

- **Sasikumar et al. (2025), “Data-driven pit stop decision support for Formula 1 using deep learning models”**, `Papers/frai-08-1673148.pdf`: es el precedente más directo por objetivo y uso de FastF1. Compara Bi-LSTM, GRU, TCN-GRU, InceptionTime y CNN-BiLSTM; su Bi-LSTM usa tres capas de 256, 128 y 64 unidades y secuencias de diez pasos. Reporta precision, recall y F1 y aplica SMOTE y class weights. Sirve como evidencia de que arquitecturas recurrentes se han probado en esta tarea, pero sus resultados no son comparables directamente con nuestro F2: cambian tamaño, balanceo, definición/protocolo de datos y arquitectura; además, cualquier modelo bidireccional exige demostrar que su entrada está truncada al instante de decisión para evitar usar futuro.
- **Mistry y Smith, “AutoF1: An RNN-Based Approach to Simulating Strategic Decision-Making in Formula 1”**, `Papers/PBRX_F1_IEEE_Conference_Version.pdf`: estudia una simulación estratégica y compara RNN con LSTM/GRU en escenarios autoregresivos. Es útil como antecedente de secuencias para estrategia, pero no establece que LSTM gane en nuestra etiqueta, cohorte o protocolo.
- **“Strategic and Environmental Variables as Predictor”**, `Papers/Strategic_and_Environmental_Variables_as_Predictor.pdf`: utiliza xLSTM, Transformer, Random Forest y XGBoost para predecir posición final en cuatro clases sobre datos de OpenF1. El objetivo y la fuente difieren; xLSTM no equivale a la LSTM estándar propuesta aquí.

Por tanto, estos papers motivan probar la arquitectura, no escogerla por resultados publicados.

## Diseño ortogonal

| Elemento | Referencia | Candidata |
|---|---|---|
| Celda secuencial | RNN tanh, 32 unidades | LSTM unidireccional, 32 unidades |
| Capas recurrentes | 1 | 1 |
| Rama estática FFNN | 16 unidades | Igual |
| Cabeza de fusión FFNN | 16 unidades | Igual |
| Dropout | 0,0 | 0,0 |
| Datos, máscara, W, preprocesador y semilla | Congelados | Idénticos |
| Optimizador, BCE, learning rate, batch y presupuesto | Fijos | Idénticos |
| Selección | F2 a 0,5 para checkpoint; después umbral F2 | Igual |

Se entrenan ambos desde cero en la misma sesión GPU de Colab, con la misma semilla y entorno. La referencia RNN se ejecuta junto a la LSTM como control pareado; se comprueba si reproduce los pesos/probabilidades de la referencia ya publicada. Si no coincide, se conserva la comparación nueva, se explica la diferencia y no se atribuye todo el delta a la arquitectura sin investigar.

Mantener 32 unidades no iguala el número de parámetros: una LSTM necesita más matrices y puertas que una RNN. La comparación estima el cambio de arquitectura a ancho fijo, incluyendo el aumento de parámetros que esa arquitectura implica. Se reportarán parámetros, tiempo, F2, F1, precision, recall, AP, BCE, alertas, matriz de confusión y cobertura por evento; no se presentará como comparación a capacidad igualada.

## Evaluación y límites

Train será 2022–2023 y validación 2024. El holdout 2025 no se empaqueta ni se consulta. Early stopping conserva la regla acordada: F2 de validación a umbral fijo 0,5, con BCE como desempate. Con el checkpoint congelado, se selecciona el umbral F2 en la cuadrícula existente. Como validación participa en ambas selecciones, el resultado es de desarrollo, no una estimación final independiente.

La métrica primaria es F2. La LSTM solo será preferible en esta pregunta si mejora F2 bajo el protocolo acordado y la interpretación de sus diferencias en eventos y falsas alertas es aceptable. Una sola semilla no demuestra estabilidad. Si el resultado es prometedor, el paso posterior sería una repetición planificada con varias semillas, no añadir dropout, cambiar el ancho o ajustar la tasa de aprendizaje dentro de esta comparación.

## Contrato de implementación

- Reutilizar loader, entradas, transformaciones, entrenamiento, selección F2, exportación y evaluación existentes.
- Añadir la celda LSTM sin modificar el camino numérico de la RNN ni romper la lectura de checkpoints previos.
- Para slots inactivos, conservar simultáneamente `h_t` y `c_t`; inicializarlos en cero por muestra.
- Hacer que el cargador de evaluación reconstruya la arquitectura desde `model_spec.json` y el hash del código empaquetado.
- Empaquetar solo train/validación, fuentes, configuración y dependencias necesarias. Exigir CUDA; no permitir fallback a CPU durante entrenamiento.
- La evaluación local usa CPU y exige paridad con probabilidades de Colab dentro de las tolerancias declaradas.
- No entrenar aquí ni consultar test. El notebook debe generar ZIP reproducible, registrar SHA-256 y detenerse ante GPU ausente o artefactos existentes.

Implementación disponible: `src/f1_pitstop/lstm_model.py`, `scripts/entrenar_comparacion_lstm.py`, `scripts/empaquetar_experimento_lstm.py`, `configs/experimento_lstm_rnn_ref_v1.json`, `configs/experimento_lstm32_v1.json` y `notebooks/05_rnn_lstm_colab.ipynb`. El código RNN previo y sus hashes se conservaron para compatibilidad. El árbol de linaje incluye ahora la rama LSTM.
