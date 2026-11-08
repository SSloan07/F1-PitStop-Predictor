# Preparación de selección F1 en LSTM32 dropout 0,2

**Estado:** listo para empaquetar y ejecutar en Colab GPU; la pareja F1/F2 aún no se ha entrenado.

La referencia y candidata conservan LSTM32, dropout 0,2, BCE, Adam con tasa fija 0,001 y el split vigente. Solo cambia el objetivo de selección de checkpoint y umbral F2→F1. También se calculó por separado el control de umbral F1 sobre los pesos F2 congelados, sin reentrenar.

- Especificación: [selección F1 frente a F2](../specs/18_experimento_lstm_f1.md).
- Notebook Colab: [`08_lstm_f1_colab.ipynb`](../../notebooks/08_lstm_f1_colab.ipynb).
- Configuraciones: `configs/experimento_lstm32_f1_ref_v1.json` y `configs/experimento_lstm32_f1_candidate_v1.json`.
- Control de umbral sin entrenamiento: [`control.json`](evidencia_control_umbral_lstm_f1_v1/control.json).

Generar el ZIP desde la raíz con `python scripts/empaquetar_lstm_f1.py --output runs/lstm_colab_f1_v1.zip`, guardar el SHA-256 y subirlo a Colab con GPU. Devuelve el ZIP de salida intacto `lstm32_f1_objective_v1_seed42.zip` a `runs/` para verificar hashes y paridad en CPU. La prueba 2025 permanece reservada.
