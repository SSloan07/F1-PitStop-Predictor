# Experimento pareado: dropout en LSTM32

**Estado:** ejecutado en Colab y reproducido localmente en CPU. Dropout 0,2 elevó F2 en esta semilla; la prueba 2025 permanece reservada. Ver [resultados](../aclaraciones/18_resultados_lstm_dropout.md).

## Pregunta

¿Dropout 0,2 después de ReLU en la cabeza de fusión mejora F2 de la LSTM32 causal frente a la misma LSTM32 sin dropout?

## Diseño ortogonal

Se entrenan dos modelos desde cero en una sesión GPU: referencia LSTM32 con dropout 0,0 y candidata LSTM32 con dropout 0,2. El factor único es `model.dropout_rate` en la cabeza de fusión después de ReLU. No se agrega dropout recurrente ni cambia la LSTM, número de unidades, FFNN, entradas, ventanas, máscaras, preprocesador, semilla ni entrenamiento.

| Control | Referencia | Candidata |
|---|---:|---:|
| Celda recurrente causal | LSTM, 32 unidades | Igual |
| Dropout después de ReLU | 0,0 | 0,2 |
| Ventana y máscara | W=8, controles causales | Igual |
| Train / validación | 2022–2023 / 2024 | Igual |
| Semilla / optimizador / pérdida | 42 / Adam / BCE sin ponderación | Igual |
| Selección | F2 a 0,5 para checkpoint; umbral F2 posterior | Igual |
| Dispositivo | CUDA obligatoria | Misma GPU y sesión |

El runner rechaza diferencias adicionales de configuración y no autoriza el bundle si hardware, versiones o poblaciones divergen. Un mismo seed no implica idéntica trayectoria estocástica con dropout activo.

## Métrica y límites

F2 de la clase positiva es primaria. Se reportan F1, precisión, recall, AP, BCE, matriz de confusión, alertas y cobertura por evento. Early stopping conserva máximo 30 épocas, paciencia 6 y desempate por BCE menor. La validación interviene en checkpoint y umbral, así que sus resultados son de desarrollo. No empaquetar ni evaluar 2025. Una sola semilla no establece estabilidad; dropout no se cambiará dentro de esta corrida.

## Ejecución

Generar desde la raíz: `python scripts/empaquetar_lstm_dropout.py --output runs/lstm_colab_dropout_v1.zip`. Registrar el SHA-256, subir el ZIP y ejecutar `notebooks/06_lstm_dropout_colab.ipynb` con GPU en Colab. El notebook valida hash y manifiesto y se detiene si CUDA no está disponible. La salida es `lstm32_dropout02_f2_v1_seed42.zip`; devolverla sin modificar a `runs/`.

Después se evalúan ambos hijos localmente con CPU, sus umbrales congelados y el pipeline existente. La evaluación debe comprobar los hashes de los artefactos y la paridad de probabilidades antes de comparar. No se reentrena ni se reajusta umbral localmente.

## Resultado observado

La pareja LSTM32 con dropout 0,0/0,2 obtuvo F2 de validación 0,43737/0,44387 con umbrales 0,08/0,05. Dropout aumentó recall 5,83 puntos porcentuales y cubrió 40 eventos adicionales, junto con 1.107 falsas alertas más; Average Precision bajó 0,01157 y BCE mejoró 0,02110. El umbral candidato quedó en el límite inferior de la cuadrícula. Es una comparación de una semilla; el informe detalla la interpretación y la comparación contextual con RNN32.

El ZIP tuvo SHA-256 `97ff3d0fea6306c5e700c0a94c8fd807814f63448edf7f18e95939d9177aa1d3`; los 42 hashes del manifiesto pasaron. La CPU reprodujo ambos hijos dentro de tolerancia con cero cambios de decisión. El test 2025 no se utilizó.

Referencias: [diseño ortogonal](05_diseno_ortogonal_experimentos.md), [contrato de preprocesamiento](10_contrato_preprocesamiento.md), [resultados LSTM32 sin dropout](../aclaraciones/16_resultados_lstm_rnn32.md).
