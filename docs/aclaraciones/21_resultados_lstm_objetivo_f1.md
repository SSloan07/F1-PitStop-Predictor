# Comparación pareada: selección por F1 frente a F2 en LSTM32

**Estado:** completada. **Datos evaluados:** validación 2024. **Prueba 2025:** reservada y no utilizada.

## Diseño

Se compararon dos entrenamientos desde cero de la misma LSTM32 causal con dropout 0,2, BCE sin ponderación, semilla 42 y CPU. La única diferencia declarada fue el objetivo usado para seleccionar el checkpoint y el umbral: F2 en la referencia y F1 en la candidata. Por tanto, este experimento compara procedimientos de selección; no compara funciones de pérdida F1 y F2. Ambos umbrales se eligieron en la misma validación, así que las métricas informadas son de desarrollo, no una estimación independiente.

Artefacto de origen: [ZIP de la pareja](../../runs/lstm32_f1_objective_cpu_v1_seed42.zip). SHA-256 registrado en la publicación del árbol. La pareja quedó marcada como completada y autorizada, con `test_used=false`.

## Resultados en validación al umbral seleccionado

| Selección | Época | Umbral | F1 | F2 | Precisión | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F2 (referencia) | 20 | 0,08 | 0,2865 | **0,4528** | 0,1777 | **0,7387** | 7.911 | **605** |
| F1 (candidata) | 17 | 0,32 | **0,3108** | 0,3936 | **0,2302** | 0,4786 | **3.706** | 1.207 |
| Diferencia F1 − F2 | — | — | +0,0243 | −0,0592 | +0,0524 | −0,2600 | −4.205 | +602 |

Con umbral fijo 0,5, F1 pasa de 0,2615 a 0,2655 y F2 de 0,2823 a 0,2815. El AP cambia de 0,2013 a 0,2019. La mayor diferencia aparece al aplicar los criterios de selección y los umbrales resultantes: F1 reduce alertas falsas y también cubre menos positivos.

## Lectura y límites

Para el objetivo del proyecto —priorizar la detección de paradas próximas— F2 conserva la ventaja en esta corrida: mayor recall y F2, a cambio de más falsos positivos. La selección F1 ofrece un punto operativo más conservador, con menos falsas alarmas y más eventos no detectados. La corrida no demuestra que una estrategia sea universalmente superior: corresponde a una sola semilla y una partición de validación.

No se reporta cobertura por evento en esta publicación porque el paquete del experimento no contiene esa métrica ni los identificadores que permitirían reconstruirla de forma verificable. No se usó prueba 2025 para escoger modelos o umbrales. Mantener F2 como objetivo principal sigue siendo coherente con la prioridad de recall; F1 queda como alternativa de operación si el costo de falsas alertas aumenta.
