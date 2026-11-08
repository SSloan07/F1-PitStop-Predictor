# Resultados: scheduler de learning rate en LSTM32 dropout 0,2

**Estado:** pareja ejecutada en Colab y reproducida localmente en CPU. El scheduler no llegó a reducir la tasa; ambas corridas conservaron el learning rate fijo y produjeron los mismos pesos y resultados. El test 2025 permanece reservado.

## Integridad y controles

El ZIP `runs/lstm32_scheduler_f2_v1_seed42.zip` tiene SHA-256 `5bc1f86441e2702d51e9d944d921626957f4bb68f3e6f61c3cc463d14b093c69`. Los 42 archivos del manifiesto pasaron sus hashes; el paquete declara `comparison_authorized=true` y `test_used=false`. Ambos hijos usaron LSTM32 con dropout 0,2, Adam, tasa inicial 0,001, semilla 42, la misma sesión T4 y versiones de software. El único cambio declarado fue la política de scheduler.

La candidata usó `ReduceLROnPlateau` sobre BCE de entrenamiento, con factor 0,5, paciencia 2, umbral relativo 0,0001 y mínimo 0,00001. La tasa permaneció en 0,001 durante las 30 épocas: el scheduler no se activó. Los pesos de ambos hijos tienen el mismo SHA-256 (`434307934cb256352736481e76ed0ae2f64e439b52e9a88dd3535c0a55816d0c`), que también coincide con los pesos de la corrida anterior LSTM32 dropout 0,2.

La evaluación CPU reprodujo las predicciones exportadas con diferencia máxima de `5,07×10⁻⁷` para ambos hijos y cero cambios de decisión. Las predicciones de referencia y candidata son idénticas entre sí. Se evaluaron 25.635 muestras elegibles de 2024 y 824 eventos con oportunidad evaluable; no se usó 2025.

## Resultado principal

| Validación 2024 | Adam, tasa fija 0,001 | Adam + `ReduceLROnPlateau` | Δ scheduler − fija |
|---|---:|---:|---:|
| Tasa usada durante las épocas | 0,001 | 0,001 | Sin reducción |
| Umbral F2 seleccionado | 0,05 | 0,05 | 0 |
| **F2** | 0,44387 | 0,44387 | 0 |
| F1 | 0,27012 | 0,27012 | 0 |
| Precisión | 16,35 % | 16,35 % | 0 pp |
| Recall por muestra | 77,71 % | 77,71 % | 0 pp |
| Average Precision | 0,19731 | 0,19731 | 0 |
| BCE | 0,38104 | 0,38104 | 0 |
| TP / FP / FN / TN | 1.799 / 9.206 / 516 / 14.114 | 1.799 / 9.206 / 516 / 14.114 | 0 |
| Eventos cubiertos | 682 / 824 (82,77 %) | 682 / 824 (82,77 %) | 0 |

Ambos seleccionaron la época 27 de 30 y tienen 12.377 parámetros. El resultado no muestra un efecto del scheduler porque la condición adaptativa nunca cambió la tasa. No debe interpretarse como evidencia de equivalencia entre tasa fija y schedulers en general: solo indica que esta configuración de `ReduceLROnPlateau`, con este monitor, paciencia y trayectoria de entrenamiento, no se activó.

## Lectura y siguiente decisión

La curva de BCE de entrenamiento no presentó un estancamiento suficientemente largo para activar el scheduler. Por eso el modelo candidato siguió la misma trayectoria que la referencia y reprodujo sus pesos exactamente. La validación tampoco mejora ni empeora, y no se justifica cambiar el baseline a partir de este experimento.

Para estudiar una política adaptativa que sí pueda actuar, el próximo diseño debe cambiar un solo detalle previamente acordado —por ejemplo, monitorizar una señal distinta o ajustar la paciencia del scheduler— y registrar de antemano qué activa la reducción. No conviene escoger esos valores mirando el resultado de validación ya observado. También se puede cerrar esta rama y conservar la tasa fija actual como referencia.

## Artefactos y reproducción

- Corrida Colab: `runs/lstm32_scheduler_f2_v1_seed42.zip`.
- Corridas extraídas: `runs/lstm32_scheduler_f2_v1_seed42/{lstm32_reference,lstm32_scheduler}/`.
- Evaluaciones CPU: `eval/lstm32_scheduler_f2_v1_seed42/{lstm32_reference,lstm32_scheduler}/`.
- Comparativo y manifiesto: `eval/lstm32_scheduler_f2_v1_seed42/comparison/`.
- [Especificación del experimento](../specs/17_experimento_lstm_scheduler.md).
- [Árbol de experimentos](linaje_experimentos_v1/publicacion_lstm_scheduler/arbol.html).

La evaluación local no reentrenó, no reajustó el umbral y no consultó el test. No hice commit ni push.
