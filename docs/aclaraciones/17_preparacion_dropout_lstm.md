# Preparación de dropout en LSTM32

**Estado:** experimento ejecutado y evaluado. Dropout 0,2 elevó F2 frente a la LSTM sin dropout en la comparación pareada de semilla 42; aún no demuestra una mejora estable entre semillas.

El diseño compara dos LSTM32 causales en la misma sesión GPU: referencia con dropout 0,0 y candidata con 0,2 después de ReLU en la cabeza de fusión. Solo cambia ese factor. Train usa 2022–2023, validación 2024 y 2025 permanece reservada. El contrato está en la [especificación](../specs/16_experimento_lstm_dropout.md).

## Artefactos

- Paquete Colab: `runs/lstm_colab_dropout_v1.zip`.
- SHA-256: `6f1ea14a7fe22808d30445f4f01655db188bdf39b7d503bc35264bd428b0f71d`.
- Notebook: [`06_lstm_dropout_colab.ipynb`](../../notebooks/06_lstm_dropout_colab.ipynb).
- Configuración de referencia y candidata: `configs/experimento_lstm32_dropout_ref_v1.json` y `configs/experimento_lstm32_dropout_02_v1.json`.

## Ejecutar y devolver

Subir el ZIP a Colab, seleccionar GPU y pegar el SHA-256 en `EXPECTED_PACKAGE_SHA256` de la primera celda. Ejecutar las celdas en orden desde un runtime limpio. El notebook falla si CUDA no está disponible o si el paquete no coincide.

El ZIP recibido pasó la validación de hashes y su evaluación se reprodujo en CPU. La comparación, el resultado y sus límites están documentados en [Resultados: dropout en LSTM32 causal](18_resultados_lstm_dropout.md). La evaluación no reentrenó, no reajustó umbrales y no consultó 2025.

La integridad del paquete local pasó para 21 archivos; el ZIP de corrida pasó 42 hashes. No incluye arrays de test. La paridad de inferencia CPU pasó sin diferencias de decisión. No se hizo commit o push.
