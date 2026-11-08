# Experimento pareado: selección F1 frente a F2 en LSTM32 dropout 0,2

**Estado:** preparado para Colab GPU; no se ha entrenado esta pareja. El test 2025 permanece reservado.

## Pregunta

En la misma LSTM32 causal con dropout 0,2, ¿cambia el resultado al seleccionar checkpoint y umbral por F1 en lugar de F2?

## Diseño ortogonal

Se entrenan desde cero dos modelos en la misma sesión CUDA. La referencia utiliza el protocolo F2 vigente; la candidata cambia `experiment_id` y `selection.primary_metric` a F1. El código mantiene BCE sin ponderación como pérdida, el mismo Adam, learning rate fijo 0,001, early stopping de seis épocas, umbral fijo 0,5 para checkpoint selection, y la misma cuadrícula de umbrales para la selección posterior. No se usa scheduler, ya que el experimento previo nunca redujo la tasa.

Se fijan LSTM causal de 32 unidades, FFNN, dropout 0,2 después de ReLU, datos, particiones temporales, preprocesador, entradas, W=8, máscaras, semilla 42 y entorno T4. La selección de checkpoint cambia de F2@0,5 a F1@0,5 y el umbral congelado cambia de maximizar F2 a maximizar F1; ambos forman el factor experimental de objetivo de selección, no cambios en la pérdida BCE. El runner rechazará diferencias adicionales.

| Control | Referencia | Candidata |
|---|---|---|
| Arquitectura / regularización | LSTM32 / dropout 0,2 | Igual |
| Pérdida / optimizador / tasa | BCE / Adam / 0,001 fija | Igual |
| Selección de checkpoint | F2 a umbral 0,5 | F1 a umbral 0,5 |
| Selección posterior de umbral | F2 en cuadrícula congelada | F1 en la misma cuadrícula |
| Train / validación / test | 2022–2023 / 2024 / 2025 reservada | Igual |
| Semilla / GPU | 42 / CUDA T4 | Misma sesión |

Una semilla no demuestra estabilidad. La selección y el umbral usan validación 2024; por eso sus métricas son de desarrollo. No se empaqueta ni evalúa test 2025.

## Control separado de umbral

Para aislar el cambio de umbral, se aplicó F1 a los pesos LSTM32 dropout 0,2 ya congelados que fueron seleccionados por F2, sin reentrenar. En validación, el umbral cambia de 0,05 a 0,22; F1 cambia de 0,27012 a 0,29841 y F2 de 0,44387 a 0,41270. Este es un control de decisión sobre el mismo conjunto de probabilidades, no una corrida de entrenamiento. La evidencia reproducible está en `docs/aclaraciones/evidencia_control_umbral_lstm_f1_v1/control.json`.

## Ejecución

Desde la raíz: `python scripts/empaquetar_lstm_f1.py --output runs/lstm_colab_f1_v1.zip`. Registra el SHA-256, sube el ZIP a Colab y ejecuta `notebooks/08_lstm_f1_colab.ipynb` con GPU. Devuelve `lstm32_f1_objective_v1_seed42.zip` intacto a `runs/`; se validarán hashes y se reproducirá la evaluación local por CPU, sin reentrenar ni reajustar después.
