# Experimento pareado: profundidad de la LSTM32

**Estado:** pareja ejecutada en Colab CPU; revisión de resultados en [26_resultados_lstm32_dos_capas.md](26_resultados_lstm32_dos_capas.md).

**Pregunta:** ¿apilar una segunda capa recurrente mejora F2 de validación frente a la LSTM32 causal actual?

La referencia usa una `LSTMCell` de 32 unidades. La candidata usa dos `LSTMCell` unidireccionales de 32 unidades, y transmite la secuencia oculta de la primera capa a la segunda en cada vuelta. El FFNN estático, embeddings, fusión, dropout 0,2 en la cabeza, máscara causal, ventana W=8 y salida binaria permanecen iguales. No se aplica dropout entre capas. Ambas condiciones entrenan desde cero con BCE normal y selección por F2, seed42 y CPU.

La comparación es un cribado de una semilla. Si la candidata mejora, repetir con semillas pareadas antes de afirmar estabilidad. La prueba 2025 permanece reservada.

Notebook: [`11_lstm32_depth_f2_colab_cpu.ipynb`](../../notebooks/11_lstm32_depth_f2_colab_cpu.ipynb). ZIP: `runs/lstm_colab_lstm2layer_f2_cpu_v2.zip`.

SHA-256 del ZIP: `9ef2e5ca476b5e51edaef1e7245c02a4905bf30489dccda738bc1c260aaca6f8`.


Corrida: `runs/lstm32_lstm2layer_f2_cpu_v1_seed42.zip`. SHA-256: `99e372b89d994146bceebd00c07c78b5e77b748b6dfba39305d9980e44b7d8db`.
