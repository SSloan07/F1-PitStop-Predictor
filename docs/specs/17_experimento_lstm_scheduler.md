# Experimento pareado: scheduler de learning rate en LSTM32 dropout 0,2

**Estado:** ejecutado en Colab y reproducido localmente en CPU. El scheduler no redujo la tasa; los pesos y resultados de ambos hijos son idénticos. La prueba 2025 permanece reservada. Ver [informe de resultados](../aclaraciones/20_resultados_lstm_scheduler.md).

## Pregunta

¿ReduceLROnPlateau, aplicado a la LSTM32 con dropout 0,2, mejora F2 frente al mismo modelo entrenado con tasa constante?

## Factor y controles

Se cambia únicamente la política de tasa de aprendizaje. Ambos hijos empiezan con Adam y `learning_rate=0.001`; la referencia mantiene esa tasa y la candidata usa `torch.optim.lr_scheduler.ReduceLROnPlateau` con `factor=0.5`, `patience=2`, `threshold=0.0001` relativo, `cooldown=0` y `min_lr=0.00001`. El scheduler monitoriza BCE evaluada sobre train, sin dropout activo en la evaluación. Así evitamos usar repetidamente validación 2024 para ajustar la tasa y luego reportarla como métrica.

Se mantienen iguales LSTM causal de 32 unidades, dropout 0,2 después de ReLU, FFNN, datos y hashes, split temporal, W=8, máscaras, preprocesador, semilla 42, batch 256, BCE sin ponderación, clipping, máximo 30 épocas, early stopping por F2 de validación con paciencia 6 y selección posterior de umbral en la cuadrícula congelada. La comparación se ejecuta en una misma sesión CUDA. El runner rechazará cualquier diferencia adicional.

| Parámetro | Referencia | Scheduler |
|---|---:|---:|
| Optimizador | Adam | Adam |
| Tasa inicial | 0,001 constante | 0,001 |
| Scheduler | ninguno | ReduceLROnPlateau |
| Monitor del scheduler | — | BCE train en modo min |
| Factor / paciencia / mínimo | — | 0,5 / 2 épocas / 0,00001 |

El scheduler puede hacer que las trayectorias diverjan; un seed común controla la inicialización, pero no implica gradientes idénticos. Se guardarán tasa usada y tasa siguiente por época para inspeccionar cuándo se activó.

## Evaluación

Métrica primaria: F2 positiva en validación 2024, con checkpoint seleccionado por F2 a umbral 0,5, desempate por BCE y early stopping sin cambios. Con los pesos congelados se seleccionará el umbral en la cuadrícula 0,05–0,95 ya acordada; no se ampliará después de ver resultados. Se reportarán F1, F2, precisión, recall, AP, BCE, alertas falsas y cobertura por evento. La validación es desarrollo, no test independiente. El test 2025 no se empaqueta ni se consulta.

## Ejecución

Desde la raíz: `python scripts/empaquetar_lstm_scheduler.py --output runs/lstm_colab_scheduler_v1.zip`. Registrar SHA-256 y subir el ZIP a Colab. Ejecutar `notebooks/07_lstm_scheduler_colab.ipynb` con GPU. Descargar `lstm32_scheduler_f2_v1_seed42.zip` intacto y ponerlo en `runs/`. La evaluación local se hará con el pipeline CPU existente, congelando pesos y umbrales, y comprobando hashes y paridad antes de comparar.


## Resultado observado

La tasa de la candidata se mantuvo en 0,001 durante las 30 épocas; `ReduceLROnPlateau` no se activó. Ambos hijos seleccionaron época 27 y obtuvieron F2=0,44387 con umbral 0,05. Sus hashes de pesos coinciden y la reproducción CPU presentó cero diferencias de decisión. Este resultado caracteriza esta configuración, que no produjo reducción efectiva; no demuestra equivalencia general entre schedulers y tasas fijas. El ZIP Colab tiene SHA-256 `5bc1f86441e2702d51e9d944d921626957f4bb68f3e6f61c3cc463d14b093c69`.
