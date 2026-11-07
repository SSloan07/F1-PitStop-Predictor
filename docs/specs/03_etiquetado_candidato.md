# Especificación exploratoria: evidencias de cambio de neumáticos

**Estado:** propuesta para revisión y validación; no autoriza generar etiquetas definitivas. Complementa las especificaciones 01 y 02.

**Contrato de ejecución:** [Especificación 06: pipeline de etiquetado candidato](06_pipeline_etiquetado_candidato.md). Define entradas, etapas, revisión acotada, controles y entregables para el modelo implementador. Para esa entrega, la revisión de reglas se limita a 2022–2023; se conserva aquí el contexto exploratorio previo.

## Objetivo y separación de etapas

Reconstruir, con toda la información histórica, si una entrada a boxes corresponde a un cambio de neumáticos. Después asignar a cada predicción al inicio de la vuelta N una etiqueta que indique si existe un cambio en N, N+1 o N+2. El modelo no recibe estas evidencias futuras, `PitInTime`, `PitOutTime` ni la etiqueta.

Primero se clasifican **eventos**, luego se construyen **etiquetas por horizonte**. Una entrada y su salida posterior representan un solo evento, asociado a la vuelta de entrada. Una salida sin entrada identificable no se considera automáticamente una parada; la salida al inicio de la carrera no es una parada de carrera por sí sola.

## Evidencia candidata para cada entrada

- Entrada y salida: identificar el paso por boxes y su secuencia temporal. La ausencia de salida no demuestra que no hubo cambio.
- `Stint` antes y después: aumento como señal candidata; investigar tanto el aumento inmediato como el observado con retraso. No asociar un aumento distante automáticamente a una entrada cuando exista otra entrada, una interrupción o información contradictoria entre ambos.
- `Compound`, `TyreLife` y `FreshTyre`: describir el juego antes y después. El mismo compuesto o un juego usado no descartan cambio; la edad o la condición de nuevo tampoco lo confirman por sí solas.
- `TrackStatus`, estado de sesión y calidad (`FastF1Generated`, `IsAccurate`): contexto y advertencias. El estado registrado para toda una vuelta no demuestra el estado exacto en la entrada.
- Campos internos como `TyresNotChanged` y `NumberOfPitStops`: solo candidatos de investigación; disponibilidad y significado no validados. No tratar su nombre como evidencia concluyente.

Estas señales pueden derivar de las mismas fuentes y procesos de FastF1; no se deben considerar verificaciones independientes sin comprobar su procedencia.

## Estados propuestos del evento

1. **Cambio identificado:** evidencia validada y coherente de que el paso por boxes incluyó cambio de neumáticos. La combinación de entrada, salida y transición de stint es una hipótesis candidata, pendiente de contraste independiente antes de adoptarla.
2. **Sin cambio identificado:** evidencia validada de que el piloto pasó por boxes sin cambio. Una transición de stint ausente inmediatamente después no basta para concluirlo.
3. **Evidencia insuficiente:** faltantes, contradicciones o falta de una regla validada para resolver el caso.

Mientras no se contraste la hipótesis, estas categorías describen el diseño, no decisiones ya verificadas sobre eventos individuales. Se registrarán señales, motivo, fuente y versión de la regla para poder auditar cada clasificación.

## Etiqueta del horizonte

- `True`: al menos un cambio identificado ocurre en N, N+1 o N+2.
- `False`: se puede establecer que no ocurrió ningún cambio en todo el horizonte, aplicando la regla acordada para el final de carrera y la cobertura de los datos.
- Desconocida: no hay cambio identificado y falta evidencia para descartar uno; por ejemplo, un evento ambiguo o un hueco relevante. No sustituir por `False`.

Un evento confirmado es suficiente para `True` aunque otro punto del horizonte sea ambiguo. El manejo de abandono y final de carrera, así como la cobertura necesaria para afirmar ausencia, siguen pendientes de acuerdo y validación.

## Ejemplos para validar

1. Entrada en vuelta 14, salida en 15, aumento de stint y mismo compuesto: candidato a cambio; el compuesto igual no lo descarta.
2. Entrada y salida con mismo stint inmediato, seguido de aumento: investigar la atribución temporal; no clasificar automáticamente como ausencia.
3. Entrada sin salida o sin datos posteriores: información insuficiente salvo evidencia adicional validada.
4. Entrada sin cambio documentado de forma independiente: candidato a ausencia de cambio.
5. Dos vueltas sin cambio conocido y una con evento ambiguo dentro del horizonte: etiqueta desconocida, no `False`.
6. Un cambio identificado dentro del horizonte y otra vuelta ambigua: etiqueta `True`.

## Validación necesaria antes de implementar

Seleccionar casos diversos de entrenamiento y validación para revisión contra fuentes independientes, incluyendo mismo compuesto, juegos usados, transiciones retrasadas, pasos sin cambio, bandera roja, abandono y final de carrera. Registrar fuente, vuelta de evento, clasificación y desacuerdos. Medir concordancia y cobertura por caso y documentar errores y exclusiones; no elegir reglas con base en el conjunto reservado de prueba. Acordar criterios y umbrales de aceptación antes de dar por definitiva la regla. La fuente externa, la selección y los umbrales aún están pendientes.

## Plan de validación de la regla

**Estado:** contraste inicial parcial documentado en [el informe de casos](../analisis/07_contraste_eventos_neumaticos.md); la validación general sigue pendiente. La especificación 06 establece cómo completar una revisión acotada y producir etiquetas candidatas, sin promoverlas a definitivas.

1. Seleccionar y documentar casos de las temporadas 2022–2024 que cubran cambios de compuesto, posibles cambios al mismo compuesto, transiciones de stint inmediatas y retrasadas, entradas sin transición inmediata y registros incompletos. Incluir, si se identifican, paradas sin cambio, interrupciones y abandonos. Documentar criterios y número de casos seleccionados; la muestra debe reflejar también eventos ordinarios y no solo anomalías.
2. Identificar fuentes independientes que permitan contrastar piloto, carrera, vuelta y, cuando sea posible, cambio de neumáticos. Registrar fuente, enlace, evidencia y limitaciones por caso. Un registro que solo confirme una entrada no confirma necesariamente el cambio; si la fuente no permite decidir, marcar el caso como no verificable.
3. Comparar la evidencia externa con las señales de FastF1, registrando acuerdos, contradicciones y casos ambiguos. Describir disponibilidad y procedencia de las fuentes para evitar asumir independencia de datos derivados de una misma fuente.
4. Proponer y revisar la regla de clasificación y su cobertura, incluidos los estados desconocidos. Fijar criterios de aceptación antes de considerar la validación concluida; no confundir los recuentos de una muestra dirigida con una medición representativa de precisión.
5. Una vez acordada la regla, diseñar y verificar el pipeline que genere la matriz etiquetada y registre las exclusiones. La implementación y publicación de etiquetas definitivas son etapas posteriores.

La temporada 2025 permanece reservada para prueba final y no se usará para seleccionar o ajustar la regla. La exploración de 2024 para este análisis se documentará, pues está prevista como validación de modelos; cualquier impacto sobre la independencia de esa validación se evaluará antes de entrenar. No se emplearán resultados de modelos sobre 2024 para decidir la etiqueta sin registrar el cambio y revisar el protocolo.

**Entregable:** informe de casos con selección, evidencia y enlaces, señales observadas, decisión o ambigüedad, discrepancias y propuesta de regla. Su revisión debe identificar qué queda sin resolver; no garantiza que todos los eventos históricos sean clasificables.
