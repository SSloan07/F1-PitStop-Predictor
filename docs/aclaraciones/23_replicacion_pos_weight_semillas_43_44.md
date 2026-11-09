# Réplicas del experimento BCE con `pos_weight=2`

**Estado:** paquete preparado; corridas pendientes en Colab CPU.

La pareja seed42 mostró una mejora de F2 y AP en validación 2024, pero solo una semilla. Este seguimiento repite el mismo contraste con semillas 43 y 44: BCE sin ponderar frente a `BCEWithLogitsLoss(pos_weight=2)`. En cada semilla, ambos modelos usan selección de checkpoint y umbral por F2. Se mantienen arquitectura LSTM32, dropout 0,2, datos, preprocesador, optimizador, tasa y entorno. No se incluye ni se evalúa la prueba 2025.

Ejecuta [`10_lstm_pos_weight_replicates_f2_colab_cpu.ipynb`](../../notebooks/10_lstm_pos_weight_replicates_f2_colab_cpu.ipynb) en Colab CPU. El notebook descarga un ZIP combinado con ambos pares y sus manifiestos. Interpreta estabilidad por signo y magnitud de las diferencias por semilla; dos réplicas adicionales son evidencia limitada, no una prueba estadística concluyente.
