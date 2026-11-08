# Resultados: dropout en LSTM32 causal

**Estado:** pareja ejecutada en Colab y reproducida localmente en CPU. Dropout 0,2 elevó F2 frente a la LSTM sin dropout en esta semilla. La prueba 2025 permanece reservada.

## Integridad y controles

El ZIP `runs/lstm32_dropout02_f2_v1_seed42.zip` tiene SHA-256 `97ff3d0fea6306c5e700c0a94c8fd807814f63448edf7f18e95939d9177aa1d3`. La comparación fue autorizada, declara `test_used=false` y sus 42 hashes pasaron. Ambos hijos usaron Tesla T4, misma sesión y versiones, semilla 42, LSTM32, FFNN y datos idénticos. El único factor cambiado fue dropout después de ReLU en la cabeza de fusión, de 0,0 a 0,2.

La referencia con dropout 0,0 reproduce exactamente los pesos de la candidata LSTM sin dropout de la comparación anterior. La evaluación CPU reprodujo las probabilidades exportadas con diferencias máximas de `7,45×10⁻⁷` para la referencia y `5,07×10⁻⁷` para dropout, sin cambios de decisión. Se compararon 25.635 predicciones elegibles y 824 eventos con oportunidad evaluable. No se usó test 2025.

## Resultado principal

Train contiene 46.162 muestras elegibles de 2022–2023; validación contiene 25.635 de 2024. Cada checkpoint se seleccionó por F2 de validación a umbral fijo 0,5; después de congelar pesos se eligió el umbral F2 en la cuadrícula acordada (0,05–0,95).

| Validación 2024, umbral F2 seleccionado | LSTM32, p=0 | LSTM32, p=0,2 | Δ dropout − referencia |
|---|---:|---:|---:|
| Umbral | 0,08 | **0,05** | — |
| **F2** | 0,43737 | **0,44387** | **+0,00650** |
| F1 | 0,27554 | 0,27012 | −0,00542 |
| Precisión | 17,04 % | 16,35 % | −0,70 pp |
| Recall por muestra | 71,88 % | 77,71 % | +5,83 pp |
| Average Precision | 0,20889 | 0,19731 | −0,01157 |
| BCE | 0,40214 | 0,38104 | −0,02110 |
| TP / FP / FN / TN | 1.664 / 8.099 / 651 / 15.221 | 1.799 / 9.206 / 516 / 14.114 | +135 / +1.107 / −135 / −1.107 |
| Falsas alertas por 100 decisiones | 31,59 | 35,91 | +4,32 |
| Entradas cubiertas | 642 / 824 (77,91 %) | 682 / 824 (82,77 %) | +40 (+4,85 pp) |

Con la prioridad acordada de F2, dropout 0,2 mejora la comparación LSTM pareada: el mayor recall y 40 entradas cubiertas adicionales compensan en F2 la menor precisión y las 1.107 falsas alertas adicionales. Average Precision baja, así que la mejora de F2 no representa mejor ordenamiento global de probabilidades.

El umbral seleccionado de dropout, 0,05, está en el límite inferior de la cuadrícula. El resultado es el máximo encontrado con la cuadrícula predefinida; no exploré umbrales inferiores ni cambié el protocolo después de ver las métricas.

## Lectura para el proyecto

El F2 de LSTM dropout 0,2 (0,44387) queda cerca, pero ligeramente por debajo, del F2 de la referencia RNN32 sin dropout (0,44463). Esa comparación cruzada es contexto, no la comparación ortogonal de este experimento: esta corrida aisló dropout dentro de LSTM.

Los dos modelos seleccionaron la época 27 y ejecutaron las 30 épocas disponibles. Dropout no cambia el número de parámetros (12.377). En esta semilla es una mejora para la LSTM bajo F2 y cobertura de eventos, a cambio de más falsas alertas. No demuestra estabilidad entre semillas. Antes de reemplazar la RNN32, conviene repetir la pareja en varias semillas y definir cómo evaluar el límite inferior del umbral sin reutilizar la validación de forma oportunista.

## Artefactos y reproducción

- ZIP Colab: `runs/lstm32_dropout02_f2_v1_seed42.zip`.
- Corridas: `runs/lstm32_dropout02_f2_v1_seed42/{lstm32_reference,lstm32_dropout}/`.
- Evaluaciones: `eval/lstm32_dropout02_f2_v1_seed42/{lstm32_reference,lstm32_dropout}/`.
- Comparativo conjunto: `eval/lstm32_dropout02_f2_v1_seed42/comparison/`.
- [Especificación del experimento](../specs/16_experimento_lstm_dropout.md).
- [Árbol de experimentos](linaje_experimentos_v1/publicacion_lstm_dropout/arbol.html).

La evaluación local reproduce los artefactos exportados; no reentrena, no reajusta el umbral y no evalúa test. No hice commit ni push.
