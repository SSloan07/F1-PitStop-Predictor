# Resultados: LSTM32 causal frente a RNN32

**Estado:** pareja ejecutada en Colab y reproducida localmente en CPU. En esta semilla, LSTM no mejora la métrica principal F2. La prueba 2025 permanece reservada.

## Integridad y controles

El ZIP `runs/rnn32_lstm_vs_rnn_f2_v1_seed42.zip` tiene SHA-256 `21a3d1c53514e51061c3f6d3d0588e968970c61145b6475d5221bb58e79de27e`. El manifiesto autorizó la comparación, declara `test_used=false` y sus 41 hashes de archivos pasaron. Ambos modelos usaron la misma T4, sesión, versiones, semilla 42 y cohorte. El único factor cambiado fue la celda recurrente: RNN tanh → LSTM causal unidireccional.

La referencia RNN32 reproduce exactamente el SHA-256 de pesos de la referencia pareada anterior. En la evaluación CPU, la máxima diferencia de probabilidad frente a Colab fue `5,36×10⁻⁷` para RNN y `7,45×10⁻⁷` para LSTM; ambos tuvieron cero decisiones distintas. La comparación conserva 25.635 predicciones elegibles, de las cuales 2.315 son positivas. No se usó el test 2025.

## Resultado principal

Train contiene 46.162 muestras elegibles (2022–2023) y validación 25.635 (2024). Los dos checkpoints se seleccionaron por F2 de validación a umbral fijo 0,5 y luego eligieron su umbral F2 con los pesos congelados.

| Validación 2024, umbral F2 seleccionado | RNN32 | LSTM32 | Δ LSTM − RNN |
|---|---:|---:|---:|
| Umbral | 0,12 | 0,08 | — |
| **F2** | **0,44463** | 0,43737 | **−0,00727** |
| F1 | 0,27971 | 0,27554 | −0,00416 |
| Precisión | 17,28 % | 17,04 % | −0,24 pp |
| Recall por muestra | 73,26 % | 71,88 % | −1,38 pp |
| Average Precision | 0,20158 | 0,20889 | +0,00731 |
| BCE | 0,33111 | 0,40214 | +0,07102 |
| TP / FP / FN / TN | 1.696 / 8.116 / 619 / 15.204 | 1.664 / 8.099 / 651 / 15.221 | −32 / −17 / +32 / +17 |
| Falsas alertas por 100 decisiones | 31,66 | 31,59 | −0,07 |
| Entradas cubiertas | 652 / 824 (79,13 %) | 642 / 824 (77,91 %) | −10 (−1,21 pp) |

A umbral fijo 0,5, LSTM obtuvo F2 0,33476 frente a 0,25835 para RNN. Ese es el criterio usado para elegir el checkpoint; el resultado principal, sin embargo, aplica a los umbrales F2 seleccionados después de congelar cada modelo. Bajo ese protocolo acordado, LSTM quedó por debajo de RNN.

## Capacidad y lectura

Con el mismo ancho oculto 32, LSTM tiene 12.377 parámetros frente a 4.121 en RNN (3,00×). Esta pareja evalúa el cambio de celda a ancho fijo, no a capacidad igualada. LSTM ejecutó las 30 épocas disponibles y seleccionó la época 27; RNN ejecutó 16 y seleccionó la 10. Los tiempos registrados fueron 175,7 s y 91,5 s, respectivamente; son duraciones de estas corridas, no un benchmark aislado.

LSTM mejoró AP, pero empeoró el F2 de validación, BCE y cobertura por evento. Su F2 de train al umbral seleccionado fue 0,68765 frente a 0,61683 para RNN, mientras el F2 de validación fue menor; junto con la brecha de BCE, esto es compatible con mayor sobreajuste en esta corrida, aunque una semilla no basta para afirmarlo como propiedad general de la arquitectura. La mayor capacidad y más tiempo no se tradujeron en una mejora de F2.

**Decisión para el proyecto:** conservar RNN32 sin dropout como referencia F2. Esta LSTM no la reemplaza. El resultado no demuestra que LSTM sea peor en general; indica que esta configuración de una capa, 32 unidades y semilla 42 no mejoró el objetivo principal. Antes de elegir entre arquitecturas con confianza convendría una repetición pareada de varias semillas, manteniendo fijo el protocolo.

## Artefactos y reproducción

- ZIP Colab: `runs/rnn32_lstm_vs_rnn_f2_v1_seed42.zip`.
- Corridas extraídas: `runs/rnn32_lstm_vs_rnn_f2_v1_seed42/{rnn32_reference,lstm32_candidate}/`.
- Evaluaciones: `eval/rnn32_lstm_vs_rnn_f2_v1_seed42/{rnn32_reference,lstm32_candidate}/`.
- Comparativo conjunto: `eval/rnn32_lstm_vs_rnn_f2_v1_seed42/comparison/`.
- Especificación: [experimento LSTM32 pareado](../specs/15_experimento_lstm_rnn32.md).
- [Árbol actualizado](linaje_experimentos_v1/publicacion_lstm/arbol.html).

La CPU local solo reproduce las probabilidades y métricas exportadas; no reentrena, reajusta umbrales ni evalúa test. No se creó commit ni se hizo push.
