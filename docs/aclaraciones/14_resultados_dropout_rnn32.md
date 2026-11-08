# Resultados del experimento de dropout en RNN32

**Estado:** comparación pareada completada y reproducida localmente. **No se detectó una mejora de F2** en esta semilla. La prueba 2025 continúa reservada.

## Diseño y controles

La pareja `rnn32_dropout02_f2_v2_seed42` entrenó ambos modelos desde cero en la misma sesión de Google Colab, con GPU Tesla T4, Python 3.13.16, PyTorch 2.14.1+cu130, NumPy 2.4.6 y pandas 2.2.3. El único factor cambiado fue `model.dropout_rate`, de 0,0 a 0,2 después de ReLU en la capa de fusión. La referencia RNN32 mantuvo la misma arquitectura y los mismos controles que la candidata.

Se conservaron las 46.162 muestras elegibles de train 2022–2023, las 25.635 elegibles de validación 2024, el preprocesador congelado, W=8 y la semilla 42. Early stopping seleccionó el checkpoint por F2 en validación a umbral 0,5, con desempate por BCE menor; después se congelaron los pesos y se eligió el umbral F2 en la cuadrícula vigente. Los dos modelos seleccionaron la época 10. La configuración no incluye ni evalúa 2025.

## Resultado principal

| Métrica en validación 2024, umbral seleccionado | RNN32, p=0 | RNN32, p=0,2 | Δ dropout − referencia |
|---|---:|---:|---:|
| Umbral | 0,12 | 0,14 | — |
| **F2** | **0,44463** | 0,44341 | **−0,00122** |
| F1 | 0,27971 | 0,28551 | +0,00581 |
| Precisión | 17,28 % | 17,92 % | +0,63 pp |
| Recall por muestra | 73,26 % | 70,24 % | −3,02 pp |
| Average Precision | 0,20158 | 0,20882 | +0,00724 |
| BCE | 0,33111 | 0,31991 | −0,01121 |
| TP / FP / FN / TN | 1.696 / 8.116 / 619 / 15.204 | 1.626 / 7.449 / 689 / 15.871 | −70 / −667 / +70 / +667 |
| Falsas alertas por 100 decisiones | 31,66 | 29,06 | −2,60 |
| Entradas cubiertas | 652 / 824 (79,13 %) | 635 / 824 (77,06 %) | −17 (−2,06 pp) |

A umbral fijo 0,5, dropout obtuvo F2 0,26788 frente a 0,25835, con 642 TP y 2.081 FP; la referencia tuvo 616 TP y 2.046 FP. Sin embargo, con el protocolo acordado de seleccionar umbral para F2, la candidata terminó con un F2 ligeramente menor. El umbral 0,5 es un diagnóstico común, no el criterio operativo final de esta pareja.

## Interpretación

En esta corrida, dropout mejoró la separación global según Average Precision y redujo BCE, además de producir menos falsas alertas en el punto de operación seleccionado. A cambio, perdió 70 positivos netos, redujo el recall 3,02 puntos y cubrió 17 entradas a boxes menos. **Como el criterio principal es F2, dropout 0,2 no gana esta comparación.** La RNN32 sin dropout permanece como referencia.

La diferencia de F2 es pequeña y procede de una sola semilla. No permite concluir que dropout perjudique en general, ni que las mejoras de AP/BCE se mantengan en otras semillas. No se seleccionará el modelo usando 2025. Una repetición con varias semillas podría medir estabilidad, pero constituye otro bloque de corridas.

## Integridad y evaluación local

El ZIP de entrada [`rnn32_dropout02_f2_v2_seed42.zip`](../../runs/rnn32_dropout02_f2_v2_seed42.zip), SHA-256 `96b4af9a62e08ed8472bb0b7c351f54ef81392380cbd59c54760a0c7c9e80f4e`, pasó la comprobación de archivo, los 40 hashes del manifiesto y las puertas de comparación: misma población, configuración salvo dropout, semilla, entorno CUDA y `test_used=false`.

La evaluación CPU reprodujo ambos modelos dentro de las tolerancias: diferencia absoluta máxima de probabilidad **5,37×10⁻⁷** para la referencia y **5,67×10⁻⁷** para dropout, con **cero decisiones cambiadas** en ambos. Los 824 eventos confirmados de 2024 tenían oportunidad evaluable. La prueba 2025 no se cargó ni se usó.

Artefactos locales: [comparación JSON](../../eval/rnn32_dropout02_f2_v2_seed42/comparison/comparison.json), [tabla CSV](../../eval/rnn32_dropout02_f2_v2_seed42/comparison/comparison.csv), [figura](figuras/14_rnn32_dropout_validacion_2024.png), y evaluaciones individuales en `eval/rnn32_dropout02_f2_v2_seed42/`.

La corrida fallida previa se conserva bajo el intento v1; no se sobrescribió. La v2 es el intento completo y evaluado. Ninguno de estos cambios se entrenó localmente.
