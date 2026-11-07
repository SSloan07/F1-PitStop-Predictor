# Exploración inicial: evidencia para etiquetar cambios de neumáticos

**Estado:** exploración preliminar, no establece una regla definitiva ni valida cambios reales de neumáticos.  
**Fuente:** archivos por carrera existentes en `EDA/dataset/datos_carreras_2022_2025/`, generados previamente con FastF1, sin clima ni telemetría. No se consultaron fuentes externas ni se descargaron datos adicionales.

## Alcance y procedimiento

Se leyeron los 56 archivos `AAAA_RR.pkl`: 22 de 2022, 22 de 2023 y 12 de 2024, con 61.497 filas en total. La muestra no contiene 2025 y no cubre todas las carreras planeadas. No se evaluó la integridad de cada archivo ni se verificaron los eventos frente a una fuente independiente. No usar estos recuentos como estimación de rendimiento de una etiqueta.

Se ordenaron filas por temporada, ronda, número de piloto y número de vuelta. Para cada fila con `PitInTime`, se compararon `LapNumber`, `Stint`, `Compound`, `TyreLife` y `PitOutTime` con la siguiente fila del mismo piloto en la misma carrera. Una vuelta se considera consecutiva cuando el número siguiente es exactamente el actual más uno. Se trata de una comparación descriptiva, sin inferir la causa del cambio.

## Resultados observados

- Se encontraron 2.233 filas con `PitInTime`; 2.160 tenían una fila consecutiva posterior y valores de `Stint` en ambas.
- De esas 2.160, en 2.152 el `Stint` aumentó y en 8 permaneció igual. No se observaron disminuciones en este subconjunto.
- En las 2.152 entradas con aumento de `Stint`, la vuelta siguiente tenía `PitOutTime`. En 558 el nombre del compuesto era igual antes y después; en 1.594 era diferente.
- En 73 entradas no había una fila posterior consecutiva para el piloto. Esto no identifica por sí mismo la causa: podría involucrar finalización, abandono, vueltas ausentes u otros casos.

Ejemplos descriptivos: en la ronda 1 de 2022, VER tiene `PitInTime` en vuelta 14; la vuelta 15 muestra `PitOutTime`, `Stint` 2 frente a 1 y compuesto `SOFT` en ambas. En la ronda 7 de 2022, LAT tiene `PitInTime` en vuelta 19 y el `Stint` permanece en 3 en la vuelta 20. Estos ejemplos muestran patrones de los datos, no una confirmación independiente de si hubo cambio de neumáticos.

## Interpretación y preguntas abiertas

Comparar solo el nombre de `Compound` no permite detectar todos los posibles cambios: puede mantenerse el mismo compuesto. Una entrada a boxes tampoco demuestra por sí misma que se cambiaron neumáticos. El aumento de `Stint` junto con entrada y salida es una **señal candidata**, pero todavía no se ha comprobado que sea una confirmación fiable ni que capture todos los cambios. No asumir que `FreshTyre` significa necesariamente cambio de neumáticos: su semántica debe verificarse, al igual que la de `Stint` y `TyreLife` en la versión de FastF1 utilizada.

Falta estudiar, entre otros, cambios durante bandera roja, pasos por boxes sin cambio, retiradas, huecos de vuelta, registros generados, datos inconsistentes y la asociación temporal de un evento con su vuelta. Una evidencia de cambio obtenida después del evento solo se podrá usar para etiquetar, nunca para predecir retrospectivamente.

## Próximos pasos de validación

1. Contrastar la definición de los campos con documentación de FastF1 y registrar la versión de la fuente.
2. Revisar manualmente una selección diversa de casos contra registros independientes, incluyendo cambios del mismo compuesto, entradas sin cambio aparente y paradas al final de carrera.
3. Examinar cambios de `Stint` que no tengan una entrada y salida correspondientes, y casos de bandera roja.
4. Proponer una regla con tres estados: cambio confirmado, ausencia confirmada y desconocido. Documentar su evidencia, límites y política para horizontes incompletos.
5. Formular ejemplos verificables antes de implementar la etiqueta definitiva. Mantener intacto el conjunto de prueba para la evaluación final; este análisis no selecciona variables ni modelos.

## Revisión adicional de `PitOutTime`

En los mismos 56 archivos se encontraron 2.216 filas con `PitOutTime`: 57 en la vuelta 1, sin una vuelta anterior consecutiva, y 2.159 con una vuelta anterior consecutiva. En estas 2.159, la vuelta anterior tenía `PitInTime`. De ellas, 2.152 muestran un aumento de `Stint` y 7 conservan el mismo `Stint`.

Así, la entrada y la salida juntas sirven para identificar un paso por boxes, pero los siete casos con el mismo `Stint` muestran que el par no determina por sí solo si hubo cambio de neumáticos. La salida de la vuelta 1 no debe contarse automáticamente como una parada durante la carrera. Estos resultados describen el subconjunto disponible, no garantizan que siempre exista esa correspondencia. Una parada de entrada en la vuelta N y salida en N+1 debe contarse como un solo evento y asociarse a la entrada al evaluar el horizonte; la salida sirve de evidencia histórica adicional.

## Safety Car y otras señales de neumáticos

Se revisó `TrackStatus` en la vuelta de entrada, usando los códigos 4 para Safety Car y 6 o 7 para Virtual Safety Car. Se observaron 449 entradas en vueltas que contienen 4 y 159 que contienen 6 o 7; los grupos se superponen y describen el estado en algún momento de la vuelta, no necesariamente en el instante exacto de entrada. De las 449 con código 4, 444 muestran aumento de `Stint` en la vuelta siguiente, una no y cuatro no tienen una vuelta siguiente consecutiva. Estas cifras no prueban que el Safety Car causara la parada ni que se cambiaron neumáticos. El estado de pista es contexto útil para el modelo si estaba disponible en el momento de predicción, pero no confirma por sí mismo el cambio.

Entre las 2.152 entradas con aumento inmediato de `Stint`, `FreshTyre` en la vuelta siguiente es verdadero en 1.568 y falso en 584. Por tanto, exigir `FreshTyre=True` descartaría muchos de estos patrones. `TyreLife` baja en 1.892, no baja en 214 y falta al menos uno de los valores comparados en 46. La edad no necesariamente reinicia a uno: podría tratarse de un juego usado; su significado exacto requiere verificación. Ninguna de estas señales por sí sola confirma un cambio físico.

Se encontraron seis aumentos de `Stint` en filas sin `PitOutTime`. En cinco, una entrada y salida aparecían dos y una vueltas antes respectivamente, y el compuesto en la fila del aumento era nulo; en el otro, hubo entrada dos vueltas antes, salida en la siguiente y aumento de stint una vuelta más tarde. Esto sugiere posibles retrasos o inconsistencias en la asignación de datos de neumáticos, que deben contrastarse antes de definir una regla. Por ello sería prematuro clasificar como ausencia de cambio toda entrada cuyo stint permanezca igual en la vuelta siguiente.

El contraste futuro debería considerar conjuntamente tiempos de entrada/salida, evolución de `Stint`, `Compound`, `TyreLife` y calidad de los registros; Safety Car ayuda a interpretar el contexto, no reemplaza la evidencia del cambio. Validar además que las fuentes se hayan publicado antes del instante de predicción cuando se usen como variables del modelo.

## Otras fuentes y variables auxiliares candidatas

La [documentación de tiempos de FastF1](https://github.com/theOehrly/Fast-F1/blob/main/docs/data_reference/time_explanation.rst) identifica `PitInTime` y `PitOutTime` como tiempos de entrada y salida del pit lane; su diferencia mide tiempo entre esos puntos, no la duración del cambio de neumáticos. Sirven para localizar el paso por boxes, no para confirmar el servicio realizado.

En los [datos de timing de FastF1](https://github.com/theOehrly/Fast-F1/blob/main/fastf1/_api.py) existen `NumberOfPitStops` y `TyresNotChanged`. El primero es un contador de paradas; FastF1 lo elimina al construir su tabla de vueltas y habría que investigar su disponibilidad, semántica y consistencia. Para `TyresNotChanged`, el propio código señala que su significado es incierto: se plantea como posible indicador de paradas sin cambio de neumáticos, pero no se puede tratar como confirmación. Ambos son datos internos y no se comprobó su disponibilidad ni se analizaron sus valores en este estudio.

Otras evidencias candidatas son el tiempo de sectores y de vuelta, y la telemetría de velocidad y posición alrededor de la entrada; podrían ayudar a revisar un paso por boxes, pero una vuelta lenta o una detención no demuestran un cambio. Los mensajes de dirección de carrera y el estado de sesión pueden contextualizar banderas rojas, penalizaciones y anomalías, pero no debe asumirse que registran todos los servicios. `FastF1Generated` e `IsAccurate` ayudan a detectar registros que requieren revisión; no son indicadores directos de cambio.

Los campos `Stint`, `Compound`, `TyreLife` y `FreshTyre` proceden de información relacionada de neumáticos y no deben contarse automáticamente como fuentes independientes. El [código de FastF1](https://github.com/theOehrly/Fast-F1/blob/main/fastf1/core.py) explica que la asociación de datos de neumáticos y vueltas requiere procesamiento y que un coche puede entrar sin cambiar neumáticos. Contrastar con una fuente externa de eventos o registros de la carrera antes de aceptar la regla. Las observaciones históricas posteriores al instante de predicción solo se usarán para etiquetado y auditoría.

## Comparación hasta dos vueltas después de la entrada

En las 2.233 entradas observadas, 2.152 muestran aumento de `Stint` en la vuelta consecutiva y otras cinco muestran el mismo `Stint` en la primera vuelta consecutiva y un aumento en la segunda; las cinco tienen salida en la primera. En las cinco, `Compound` en la segunda vuelta es nulo. Son candidatos para revisión, no cambios confirmados. El procedimiento mira dos filas consecutivas del mismo piloto y carrera y no descarta otras causas o eventos intermedios. La regla propuesta y sus límites se documentan en `docs/specs/03_etiquetado_candidato.md`.
