# Decisiones para el primer experimento: RNN, F2 y Colab

**Estado:** primer entrenamiento F2 en Colab y evaluación local completados; ver [resultados](06_resultados_rnn_v1_validacion_2024.md).
**Alcance:** fija la prioridad del primer experimento, la métrica y el flujo de trabajo entre Colab y la evaluación local. Los hiperparámetros y las reglas de selección quedan registrados en `configs/experimento_rnn_v1.json` y en [la especificación 11](../specs/11_experimento_rnn_inicial.md).

## Decisiones vigentes

1. El primer experimento será una **RNN sencilla, poco profunda y de complejidad contenida**. La propuesta anterior de comenzar por una MLP deja de ser el paso inicial. No se exige implementar otro modelo antes de esta RNN.
2. El objetivo principal es **maximizar F2 de la clase positiva** en el protocolo de validación. Se interpreta sobre las decisiones piloto-vuelta con etiqueta evaluable y elegibilidad acordada. Precision, recall, F1, matriz de confusión y cobertura por evento acompañan el resultado para entender falsas alertas y entradas omitidas.
3. El entrenamiento se ejecutará en un **notebook de Google Colab** reproducible desde una sesión limpia.
4. Se exportará el modelo para cargarlo en este repositorio y revisar su comportamiento mediante un **pipeline de evaluación local**.
5. Los experimentos aplicarán el principio de **ortogonalización** de la especificación 05.

Esta nota registra la prioridad RNN y la elección de F2, reflejadas ya en las especificaciones 05, 08 y 11. No cambia las características, etiquetas, particiones ni reglas de preprocesamiento.

## Base que se mantiene fija

- Ventanas/características: `muestras_causales_v1_3`, con W=8.
- Etiquetas: `etiquetado_pit_entry_v3_2022_2025`.
- Alineación y particiones: `muestras_causales_v1_3_etiquetas_v3_particiones_v1_1`.
- Preprocesamiento: `muestras_preprocesadas_v1_3`, ajustado solo con train supervisado elegible.
- Train: 2022–2023; validación: 2024; prueba final reservada: 2025.
- Contexto estático y secuencia dinámica mantienen sus contratos. El modelo deberá declarar cómo utiliza las máscaras y cómo combina las ramas; el relleno técnico cero no equivale a una medición presente.

## Protocolo experimental implementado

La configuración fija una RNN tanh de una capa y 32 unidades, embeddings de dimensión 4 y una rama estática de 16 unidades. Usa Adam con learning rate 0,001, BCE sin ponderación inicial, batch 256, semilla 42, máximo 30 épocas, paciencia 6 y recorte de gradiente a 1. La especificación 11 describe el consumo explícito de máscaras.

Se selecciona checkpoint con F2 de validación a umbral fijo 0,5, con menor BCE como desempate. Después, con esos pesos congelados, se busca el umbral en 0,05–0,95 con paso 0,01; los empates se resuelven por mayor precision y luego mayor umbral. Se conservan resultados a 0,5 y al umbral elegido para separar aprendizaje de decisión.

F2 pondera recall más que precision y depende del umbral usado para convertir probabilidades en alertas. Maximizarlo exige especificar cómo se seleccionan checkpoint y umbral en validación; no basta informar una pérdida de entrenamiento menor. La pérdida diferenciable de entrenamiento y la métrica de selección son decisiones distintas. La regla queda fijada antes de la corrida y no usa 2025.

El [contrato de F-beta](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html) permite reportar F2 con `beta=2` para la clase positiva. No se sustituirá esta métrica por accuracy ni por un promedio entre clases sin una decisión explícita. La cobertura por evento se informará por separado de F2 por muestra.

## Flujo Colab → paquete exportado → evaluación local

El notebook cargará código y artefactos identificados por versión/hash, verificará la alineación y seleccionará las muestras supervisadas elegibles. Reutilizará el preprocesador congelado; no reajustará estadísticas con validación.

La exportación incluirá:

- Pesos y definición/configuración exacta de arquitectura, incluidos vocabularios/cardinalidades y tratamiento de máscaras.
- Preprocesador congelado o una copia verificable por hash.
- Orden de entradas, esquema, W, claves de muestras y versiones/hashes de datos y particiones.
- Checkpoint seleccionado, época, criterio de selección, umbral y procedimiento con que se obtuvo.
- Configuración de entrenamiento, semillas y versiones del entorno.
- Historial de entrenamiento/validación y predicciones de validación con sample_id, score y etiqueta/elegibilidad asociada por clave.
- Manifiesto de los archivos exportados. Si se habilita reanudación, también optimizador y estados necesarios.

La [guía de guardado y carga de PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) respalda usar pesos `state_dict` con una definición compartida del modelo si se adopta ese framework. El notebook y el evaluador deben reutilizar la misma implementación del modelo, evitando duplicar lógica entre ambos entornos.

El evaluador local podrá cargar el paquete en CPU, reproducir probabilidades de validación y contrastarlas con las exportadas usando una tolerancia numérica documentada. No se exigirá igualdad bit a bit entre GPU y CPU. Aplicará el umbral congelado, calculará F2 y métricas auxiliares y permitirá revisar errores por carrera/piloto y ejemplos concretos. La prueba final permanecerá reservada hasta congelar el protocolo/modelo para esa evaluación.

## Ortogonalización

La configuración inicial y sus resultados serán el punto de referencia. Se conservarán datos, particiones, preprocesador, lista/orden de características, W y protocolo de evaluación al estudiar otro factor.

Una variación posterior declarará qué cambia y qué pregunta busca responder: capacidad de la red, entrenamiento, entradas, ventana o umbral. No se mezclarán cambios de varias capas sin identificarlos. Si se estudian interacciones, se describirá el diseño explícitamente. Los resultados incluirán la diferencia frente a la configuración de referencia y conservarán intentos fallidos.

Se distinguirá el efecto de cambiar pesos/arquitectura del de cambiar la regla de alerta. La elección de semillas y el presupuesto de comparación también quedarán fijados para evitar atribuir a una arquitectura una diferencia provocada por otro tratamiento.

Referencias locales: [ortogonalización](../specs/05_diseno_ortogonal_experimentos.md), [datos y evaluación](../specs/08_dataset_causal_modelado.md), [muestras causales](../specs/09_construccion_muestras_causales.md), [preprocesamiento](../specs/10_contrato_preprocesamiento.md).
