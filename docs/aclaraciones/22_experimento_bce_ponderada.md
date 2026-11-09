# Experimento: ponderación de positivos en BCE para F2

**Pregunta.** ¿Aumentar a 2 el peso de la clase positiva durante el entrenamiento mejora F2 de validación de la LSTM32 causal con dropout 0,2?

**Diseño.** Pareja desde cero, semilla 42, ejecución CPU. Referencia: `BCEWithLogitsLoss` sin ponderar. Candidata: `BCEWithLogitsLoss(pos_weight=2)`. Ambas conservan selección de checkpoint por F2 a umbral 0,5 y después selección de umbral por F2 sobre pesos congelados. Se mantienen fijos dataset, particiones, preprocesador, entradas/máscaras, arquitectura, dropout, Adam, learning rate, batch size, early stopping y entorno.

La pérdida de validación y el desempate de checkpoint usan BCE sin ponderar en ambas corridas para que sean comparables. El peso solo se aplica al gradiente de entrenamiento. El paquete no contiene arrays de prueba; 2025 permanece reservada.

**Interpretación previa.** pos_weight puede cambiar la representación y el orden de puntuaciones, aunque también altera la calibración. Por eso se informarán F2, AP, precisión, recall, FP/FN y curvas de entrenamiento; un cambio de umbral por sí solo no bastará para concluir que el ranking mejoró. Esta primera pareja es un cribado de una semilla. Si el resultado es favorable, repetir semillas pareadas antes de afirmar una mejora estable.

**Ejecución.** Notebook: `notebooks/09_lstm_pos_weight_f2_colab_cpu.ipynb`. Paquete de entrada: `runs/lstm_colab_pos_weight2_cpu_v1.zip`.
