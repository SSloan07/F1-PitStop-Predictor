# Especificación 02: pipeline de construcción de datos

**Estado:** borrador para revisión.  
**Alcance:** extracción, matriz maestra, calidad, etiqueta y publicación de datos para experimentos. Las reglas que dependan de observar datos reales están pendientes.

## Resultado esperado

Una matriz histórica validada y versionada, con una fila por temporada, carrera, piloto y vuelta, junto con un diccionario de datos y un informe de calidad. La matriz maestra puede contener columnas utilizadas para etiquetar; cada experimento seleccionará explícitamente sus entradas autorizadas.

El pipeline se ejecutará principalmente en local. Las versiones publicadas de la matriz y los resultados de experimentos se almacenarán en Google Drive. El entrenamiento se hará en notebooks de Google Colab. El formato de archivo y el mecanismo concreto de transferencia se decidirán después de verificar tipos y tamaños.

## Etapas

### 1. Extracción reanudable

Descargar información por carrera y guardar resultados parciales de manera que una interrupción o el límite de la API no obligue a repetir carreras completadas. Mantener separada la caché de la fuente de los datos procesados. Registrar carreras completadas, fallidas y pendientes; no declarar completo un dataset con carreras sin procesar. Los fallos se podrán reintentar sin duplicar carreras.

### 2. Matriz maestra

Integrar las vueltas con los datos pertinentes de clima, estado de pista y resúmenes de telemetría, conservando identificadores y tiempos necesarios para verificar disponibilidad. La granularidad es una fila por piloto y vuelta; las series por instante se resumirán sin multiplicar filas. Se documentará qué fuentes y columnas se incorporan y cómo se relacionan.

Las matrices históricas conservarán marcas temporales de las fuentes. Para cada predicción al **inicio** de la vuelta, el clima se alineará con la última observación registrada antes o en ese instante. El límite configurable `K_clima` (preferiblemente minutos) determina si la observación es suficientemente reciente; si no cumple, queda ausente. Registrar la antigüedad de la observación aceptada. Nunca usar observaciones posteriores al instante de predicción. La telemetría agregada de una vuelta solo puede alimentar predicciones posteriores a su cierre. La fila actual de predicción no puede exponer sus resúmenes completos. La observación inicial de carrera requiere tratamiento propio. La marca temporal de registro no demuestra la latencia real de publicación; documentar esta limitación.

### 3. Validación y limpieza

Comprobar, como mínimo, claves duplicadas, tipos, orden temporal, vueltas faltantes, valores imposibles, disponibilidad por fuente y carreras incompletas. Las decisiones sobre corrección, exclusión o conservación de valores faltantes deben ser explícitas. Producir un informe con recuentos antes y después, causas y advertencias; no eliminar registros silenciosamente.

La limpieza común no debe aprender estadísticas del conjunto de prueba. Las transformaciones que aprenden de los datos —como imputación estadística, escalado y codificación ajustada— se ajustarán solo con entrenamiento y se guardarán como artefactos del experimento.

### 4. Etiquetado histórico

Para cada instante de predicción, establecer si se produce un cambio de neumáticos en la vuelta actual o alguna de las dos siguientes. El proceso puede consultar toda la carrera histórica. `PitInTime` y `PitOutTime` ayudan a identificar entradas y salidas; no prueban por sí solos que se cambiaron neumáticos. Antes de publicar etiquetas definitivas, inspeccionar carreras y acordar evidencia y reglas para cambios confirmados, entradas sin cambio y casos ambiguos.

La etiqueta será booleana cuando pueda determinarse y tendrá un estado desconocido cuando no haya evidencia suficiente. Documentar el tratamiento de vueltas faltantes, final de carrera, abandonos y datos incompletos. Las muestras sin etiqueta válida no se usarán como ejemplos supervisados salvo que se resuelvan con una regla justificada.

### 5. Publicación y trazabilidad

Publicar en Drive una versión identificable de la matriz y su informe, con huella del archivo, versión de código y especificaciones, fuentes, carreras incluidas y excluidas, reglas aplicadas y fecha de creación. Una nueva regla de limpieza o etiquetado produce una versión distinguible. No sobrescribir silenciosamente una versión usada en experimentos.

## Separación y consumo experimental

La separación inicial acordada es 2022–2023 para entrenamiento, 2024 para validación y 2025 para prueba; se comprobará su cobertura antes de fijarla. Las carreras no deben cruzar particiones. El conjunto de prueba se reserva para la evaluación final, no para seleccionar variables, hiperparámetros ni umbral.

Cada experimento registra la versión de matriz y sus particiones y selecciona variables permitidas. La división entre estáticas y dinámicas depende del experimento; no altera la matriz maestra. Las ventanas no deben cruzar pilotos, carreras ni particiones. Las transformaciones ajustadas y sus parámetros se exportarán junto con los pesos, configuración y predicciones.

## Criterios de aceptación

- Una ejecución interrumpida puede reanudarse sin repetir carreras terminadas ni duplicar sus filas.
- Las claves de la matriz son únicas o cualquier excepción queda documentada y justificada.
- La salida incluye informes de cobertura, calidad y etiquetado, incluidos los casos sin resolver.
- Hay ejemplos verificables de horizonte, cambio confirmado, ausencia de cambio y casos ambiguos antes de aceptar la regla de etiquetado.
- La versión publicada puede identificarse por su huella y relacionarse con código, reglas y carreras procesadas.
- El experimento no recibe columnas prohibidas ni información posterior al instante de predicción.

## Estado del extractor existente

El extractor actual es un prototipo: su etiqueta usa entradas a boxes y no confirma cambios de neumáticos, y sus resúmenes por vuelta requieren alineación temporal con el inicio de la vuelta. No debe considerarse una implementación validada de esta especificación. Antes de usar su salida para entrenar, se deben resolver estas diferencias y verificarla con datos reales.

## Pendientes

Fuente y evidencia de cambios de neumáticos; disponibilidad temporal y latencia; regla de abandono y final de carrera; formato y estructura de Drive; integridad de fuentes y variables; reglas detalladas de exclusión y versiones.
