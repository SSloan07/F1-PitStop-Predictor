# Preparación de la comparación RNN32–RNN16

**Estado:** pareja ejecutada en Colab y evaluada localmente; ver [resultados](12_resultados_rnn16_pareada.md). La hipótesis fue que reducir la capacidad recurrente podría mejorar la generalización. No se entrenó localmente con datos reales.

## Cambio estudiado

Se conserva la referencia de selección **F2**, la pérdida BCE sin ponderación, semilla 42, W=8, características, máscaras, preprocesador y particiones. Cambia únicamente el tamaño del estado recurrente:

| Componente | Referencia de la pareja | Candidata |
|---|---:|---:|
| Estado oculto recurrente | 32 | 16 |
| Embeddings | 4 | 4 |
| Rama estática | 16 | 16 |
| Capa oculta de fusión | 16 | 16 |
| Parámetros con los vocabularios publicados | 4.121 | 2.233 |
| Selección de checkpoint | F2 a umbral 0,5 | F2 a umbral 0,5 |
| Selección posterior de umbral | Cuadrícula F2 acordada | Misma cuadrícula |

El número de parámetros de las conexiones de entrada a la fusión también se adapta al nuevo tamaño recurrente. No se cambia el tamaño configurado de esa capa ni el de la rama estática.

## Por qué se vuelve a ejecutar RNN32

La comparación F1 anterior cambió de T4/CUDA a CPU y de versión de Python. Para estudiar capacidad con un control de entorno, el notebook ejecuta ambos modelos en la misma sesión. La comparación principal será entre esos dos modelos nuevos; la corrida F2 histórica aporta contexto y se conserva.

Se recomienda GPU T4 cuando esté disponible. CPU también permite la comparación si ambos entrenamientos comparten entorno. El runner comprueba configuraciones y entorno y solo publica una comparación autorizada cuando pasan los controles. El máximo de 30 épocas y la paciencia de seis se mantienen; las épocas realmente ejecutadas y elegidas pueden diferir como resultado del entrenamiento.

## Ejecución

1. Abrir [03_rnn16_colab.ipynb](../../notebooks/03_rnn16_colab.ipynb) en Colab y elegir GPU si está disponible.
2. Subir `runs/rnn_colab_rnn16_v1.zip`; la primera celda detecta `/content/rnn_colab_rnn16_v1.zip` si ya está en Archivos.
3. Ejecutar las celdas en orden. El comando compartido entrena RNN32 y RNN16 y muestra el entorno y las métricas de la pareja.
4. Descargar `rnn16_vs_rnn32_f2_v1_seed42.zip` y guardarlo en la carpeta local `runs/` del proyecto.

El ZIP conjunto contiene los directorios `rnn32_reference/` y `rnn16_candidate/`, los resultados, comprobaciones de entorno y manifiesto de la pareja. Cada hijo conserva pesos, definición, preprocesador, configuración, umbral, predicciones y su propio manifiesto. Se excluyen ZIP anidados.

El entrenamiento sigue ejecutándose en Colab. La evaluación local posterior comprobará paridad de ambos hijos, F1/F2, precisión, recall, errores y cobertura por evento sobre 2024. **2025 permanece reservada.** Una semilla y una validación usada para selección no bastan para afirmar una mejora estable.

El nodo de capacidad P1 del [árbol de experimentos](09_arbol_experimentos.md) ya tiene resultados y se compara con E2, la referencia RNN32 de su misma pareja.

Contrato completo: [especificación 13](../specs/13_experimento_capacidad_rnn16.md). Configuraciones: [referencia RNN32](../../configs/experimento_rnn32_pareado_v1.json) y [candidata RNN16](../../configs/experimento_rnn16_v1.json).
