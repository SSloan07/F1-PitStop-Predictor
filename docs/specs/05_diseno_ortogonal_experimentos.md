# Especificación 05: diseño ortogonal de experimentos

**Estado:** guía de trabajo para los experimentos; los valores concretos se definirán en cada experimento.
**Alcance:** cómo aislar decisiones experimentales para que los resultados sean interpretables y reproducibles.

## Propósito

Aplicamos el principio de ortogonalización como una regla de diseño: cada decisión o grupo de cambios debe tener un propósito identificable y quedar registrado. No significa que los factores sean matemáticamente independientes ni que siempre se cambie una sola variable. Significa que evitamos cambiar muchos componentes sin poder atribuir el resultado a una causa.

Este documento complementa la especificación del pipeline de datos: allí se define cómo se construye la matriz; aquí se define cómo comparar modelos que consumen una versión de esa matriz.

## Capas de decisión

| Capa | Ejemplos de decisiones | Qué se busca aislar |
| --- | --- | --- |
| Datos y particiones | versión de matriz, carreras incluidas, partición por temporada/carrera | efecto de los datos y de la evaluación |
| Entradas | variables dinámicas, estáticas, indicadores de ausencia | aporte de cada grupo de características |
| Secuencia | longitud máxima de ventana, representación de ventanas cortas | efecto del historial disponible |
| Arquitectura | baseline tabular, RNN, ramas estática/dinámica | valor añadido de la estructura del modelo |
| Entrenamiento | semilla, optimizador, tasa de aprendizaje, regularización, épocas | estabilidad y ajuste del aprendizaje |
| Decisión | calibración, umbral de alerta | equilibrio entre alertas omitidas y falsas |

## Protocolo de comparación

1. Definir una configuración base completa y asignarle un identificador.
2. Mantener fijos matriz, particiones, métrica, presupuesto de entrenamiento y protocolo de evaluación al estudiar una decisión concreta.
3. Cambiar el factor de interés y registrar explícitamente cualquier otro cambio.
4. Si se quieren medir interacciones —por ejemplo, ventana y arquitectura— diseñar una comparación factorial pequeña y planificada; no mezclar cambios de manera accidental.
5. Seleccionar modelo, hiperparámetros y umbral usando validación. Reservar prueba para la evaluación final acordada.
6. Repetir con varias semillas cuando sea viable, reportando resultados por semilla y resumen de variabilidad.
7. Conservar resultados negativos y fallidos junto con la configuración que los produjo.

La partición debe mantener carreras completas en un solo conjunto y permanecer fija entre experimentos comparables. Cualquier modificación de datos, particiones o reglas de evaluación genera una nueva versión identificable y requiere explicar por qué sus resultados ya no son una comparación directa con la base anterior.

## Evaluación y alertas

El criterio principal acordado es **F2 de la clase positiva**. Reportar también precision, recall, F1, Average Precision, matriz de confusión y cantidad de alertas. El primer experimento usa una RNN sencilla y el protocolo de [la especificación 11](11_experimento_rnn_inicial.md): selección de checkpoint con F2 a umbral fijo 0,5 y búsqueda posterior de umbral con pesos congelados. Mostrar resultados de entrenamiento y validación bajo la misma regla de alerta permite distinguir ajuste y generalización. No cambiar el umbral sobre prueba.

Además de métricas por fila piloto-vuelta, informar cuántas paradas reales fueron cubiertas por al menos una alerta dentro del horizonte permitido. Las alertas consecutivas pueden solaparse; por ello, la métrica por muestra y la cobertura por evento responden preguntas distintas y no deben confundirse.

## Registro obligatorio por experimento

Cada notebook ejecutable debe dejar un manifiesto o sección de resultados con:

- identificador, pregunta experimental e hipótesis;
- versión y huella de la matriz, carreras y particiones;
- columnas de entrada y variables excluidas por fuga;
- ventana, arquitectura, hiperparámetros, épocas, semilla y entorno de ejecución;
- preprocesadores ajustados solo con entrenamiento y sus parámetros;
- métrica principal, métricas secundarias y regla de selección del umbral;
- pesos exportados, historial de entrenamiento, predicciones de validación/prueba y métricas;
- cambios frente a la configuración base, resultado y conclusión limitada a la pregunta estudiada.

Los notebooks de experimentos deben poder ejecutarse en Colab o Kaggle con la matriz publicada, sin depender de archivos locales ocultos. Deben guardar sus artefactos de salida de forma identificable para que la evaluación posterior pueda reconstruir qué modelo generó cada predicción.

## Criterios de aceptación

- Se puede reconstruir la configuración y las particiones de cada resultado.
- Dos experimentos comparables difieren solo en los factores declarados, o identifican el diseño que estudia sus interacciones.
- Ningún parámetro o umbral se elige usando el conjunto de prueba.
- Cada resultado incluye métricas consistentes, predicciones y artefactos suficientes para evaluación posterior.
- La pregunta, el cambio realizado y la conclusión se entienden sin depender del estado temporal del notebook.

## Relación con otros documentos

- [Problema y comportamiento de la alerta](01_problema.md): horizonte, instante de predicción y prevención de fuga.
- [Pipeline de construcción de datos](02_pipeline_datos.md): matriz, limpieza, etiquetas y particiones propuestas.
- Los criterios concretos de cada arquitectura y sus experimentos se detallarán en futuras especificaciones y notebooks.
