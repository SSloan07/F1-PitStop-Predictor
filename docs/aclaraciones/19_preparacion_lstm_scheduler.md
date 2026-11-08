# Preparación del experimento de scheduler en LSTM32

**Estado:** corrida completada y reproducida localmente en CPU. `ReduceLROnPlateau` no redujo la tasa en ninguna época; las métricas y los pesos de ambos hijos son idénticos.

Se comparará la LSTM32 con dropout 0,2 y tasa Adam constante frente a la misma configuración con `ReduceLROnPlateau`. El scheduler observará BCE de entrenamiento; el checkpoint y el umbral se seguirán seleccionando con el protocolo F2 de validación vigente. La especificación recoge los controles y parámetros.

- Contrato: [experimento pareado de scheduler](../specs/17_experimento_lstm_scheduler.md).
- Notebook: [`07_lstm_scheduler_colab.ipynb`](../../notebooks/07_lstm_scheduler_colab.ipynb).
- Configuraciones: `configs/experimento_lstm32_scheduler_ref_v1.json` y `configs/experimento_lstm32_scheduler_plateau_v1.json`.

El ZIP de entrada fue `runs/lstm_colab_scheduler_v1.zip`; el SHA-256 del resultado es `5bc1f86441e2702d51e9d944d921626957f4bb68f3e6f61c3cc463d14b093c69`. La evaluación local pasó paridad de CPU y no consultó 2025. Ver el [informe de resultados](20_resultados_lstm_scheduler.md).
