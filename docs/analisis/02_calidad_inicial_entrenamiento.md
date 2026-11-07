# Revisión inicial de datos de entrenamiento (2022–2023)

**Estado:** diagnóstico preliminar, no es un dataset limpio ni una autorización para entrenar. Se leyeron los archivos por carrera `2022_??.pkl` y `2023_??.pkl` de `EDA/dataset/datos_carreras_completos_2022_2025/`. No se consultaron ni ajustaron decisiones con 2025. Cifras correspondientes a los archivos existentes en esta revisión; falta cotejar la cobertura con calendarios y auditar carreras.

## Cobertura y claves

- 44 archivos de carrera: 22 de 2022 y 22 de 2023; 47.997 filas y 57 columnas tras concatenarlos.
- No se observaron nulos ni duplicados en la clave (`Temporada`, `Ronda`, `DriverNumber`, `LapNumber`). La unicidad no garantiza que no falten vueltas.
- Hay 1.779 filas con `PitInTime` y 1.764 con `PitOutTime`. Sus nulos indican normalmente ausencia del evento en la fila; no son simplemente datos a imputar. Todavía no identifican por sí solos cambios de neumáticos.

## Calidad y señales para investigar

- `TyreLife`: 555 nulos (1,16 %). `LapTime`: 0,56 % nulo; `Sector1Time`: 2,17 %; `SpeedFL`: 3,87 %. Hay que estudiar su distribución y causa antes de decidir limpieza o imputación.
- `FastF1Generated`: 79 filas verdaderas. `IsAccurate`: 7.037 falsas y 40.960 verdaderas. No eliminar indiscriminadamente: comprender los criterios y el efecto sobre paradas y ventanas.
- Todos los registros tienen al menos una muestra de telemetría según `Telemetry_Samples` (mínimo 84, máximo 14.910). Eso no garantiza calidad ni cobertura completa de la vuelta; una velocidad media observada de cero requiere revisión.
- Las columnas meteorológicas no presentan nulos en esta muestra y hay 2.007 filas con lluvia. La disponibilidad y el instante de cada observación se deben revisar para prevenir el uso de información futura.
- Los resúmenes de telemetría y el estado de pista corresponden a la vuelta completa: al inicio de una vuelta deben usarse únicamente datos previos o muestras disponibles en ese instante. No alimentar al modelo directamente con la fila actual sin ajustar su alineación.

## Siguientes comprobaciones

Contrastar los calendarios y cobertura por piloto, revisar vueltas faltantes y anómalas, consistencia temporal y eventos de boxes; analizar variables generadas y no precisas, calidad del clima y telemetría; y definir reglas de limpieza documentadas. Las transformaciones que aprenden parámetros se ajustarán solo con entrenamiento. El etiquetado de cambio de neumáticos sigue sin validarse, por lo que estas filas no están listas para entrenamiento supervisado definitivo.

## Investigación detallada y propuestas iniciales

### Faltantes vinculados a eventos

De las 1.857 filas sin `SpeedFL`, 1.778 tienen `PitInTime`: la asociación es fuerte y eliminar todas esas filas descartaría casi todas las vueltas de entrada de la muestra. Un valor ausente no equivale a velocidad cero. De los 270 `LapTime` ausentes, 79 son filas generadas; 91 tienen entrada y 124 salida. Las categorías pueden solaparse. Mantener los faltantes en la matriz maestra y estudiar por variable y contexto; no aplicar eliminación global.

Los 555 faltantes de `TyreLife` se concentran en tres carreras: 359 en 2022 ronda 11, 161 en 2022 ronda 7 y 35 en 2023 ronda 8. Deben revisarse sus registros de stint y fuente antes de imputar. Cualquier imputador que aprenda parámetros se ajustará solo con entrenamiento; también se puede conservar un indicador de dato ausente, sujeto a disponibilidad temporal.

### Vueltas generadas e `IsAccurate`

Las 79 filas `FastF1Generated=True` también tienen `LapTime` ausente y `IsAccurate=False`. FastF1 señala que estas filas pueden contener información limitada, interpolada o inferida. Deben identificarse y revisarse como posibles registros incompletos, no tratarse como mediciones completas.

Las 7.037 vueltas con `IsAccurate=False` incluyen las 1.779 filas con entrada a boxes y las 1.764 con salida. Según [FastF1](https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/timing_data.rst), el indicador evalúa sincronización y condiciones de la vuelta; una entrada, salida o ciertas condiciones de Safety Car impiden marcarla como precisa. Por eso no se deben eliminar todas esas filas: hacerlo retiraría eventos centrales para las etiquetas. Revisar cada variable según el uso, preservando la marca de calidad.

### Telemetría

Hay 27 filas con velocidad media cero; 26 corresponden a BOT en la ronda 18 de 2022 y una a MAG en la ronda 5 de 2022. La presencia de muestras no demuestra que sean mediciones válidas: investigar la serie original, escala y cobertura antes de imputar o descartar. La media y el máximo resumen la vuelta completa y son datos futuros si se usan en el inicio de esa misma vuelta.

### Alineación temporal del clima

Comparando `Weather_Time` con `LapStartTime` para las 47.997 filas, 47.973 observaciones meteorológicas asignadas a la vuelta tienen hora posterior a su inicio, y 24 son anteriores o simultáneas; ninguna es posterior al `Time` de fin de vuelta. Por tanto, no usar directamente el clima de la fila de la vuelta actual al inicio. La propuesta es obtener la última observación disponible a la hora de predicción, con marca temporal, y dejarla desconocida si no existe una anterior. Debe comprobarse que el reloj y la latencia de publicación sean compatibles con el escenario real.

### Vacíos de evidencia y próximos pasos

Esta investigación es descriptiva sobre 2022–2023; no determina causalidad ni reglas finales de exclusión. Verificar continuidad por piloto, calidad por carrera y pilotos afectados; revisar series originales de telemetría y clima con sus timestamps; analizar eventos y etiquetas con fuentes independientes. Acordar y probar reglas antes de producir una matriz limpia. No imputar con datos de validación o prueba ni borrar las columnas de auditoría de la matriz histórica.

## Inspección adicional de casos

No se observaron saltos mayores a una vuelta entre filas consecutivas por piloto y carrera. Esta comprobación no detecta faltantes antes de la primera fila o después de la última, ni garantiza que las filas generadas correspondan a vueltas realmente completadas.

De los 555 nulos de `TyreLife`, 303 tienen la cadena literal `None` en `Compound`; no es un valor nulo de pandas. Este significado y la regla de normalización requieren revisión.

Se inspeccionó la caché original de telemetría para BOT en Japón 2022: hay extensos tramos de velocidad y RPM cero, con acelerador 104; 26 vueltas guardadas tienen velocidad media cero. No se ha determinado la causa ni revisado el otro caso, MAG en Miami. No se debe confundir el número de muestras con su validez.

Las reglas candidatas se encuentran en `docs/specs/04_limpieza_candidata.md`; todavía no se ha transformado ni excluido ninguna fila.

## Compuesto desconocido y vueltas vecinas

Se revisaron las 369 filas cuyo `Compound` es la cadena `None` en 2022–2023. Para cada una se buscaron valores conocidos antes y después dentro del mismo piloto, carrera y `Stint`. Ninguna tiene un compuesto conocido en otra fila de ese mismo stint: no fue posible recuperar ninguno mediante esa comparación. Los desconocidos se concentran en 2022 ronda 7 (35), 2022 ronda 11 (299) y 2023 ronda 8 (35).

Esta comparación no prueba que el stint esté perfectamente asignado ni sustituye verificar entradas, salidas y eventos de bandera roja. Un compuesto de otro stint no se puede trasladar suponiendo que no hubo cambio; el piloto puede montar otro juego del mismo compuesto o cambiar durante una interrupción. La investigación futura podría contrastar fuentes independientes.

La reconstrucción retrospectiva usando datos posteriores puede ayudar a auditoría y etiquetado, pero esos valores no deben alimentar predicciones anteriores a su disponibilidad: el valor para entrenamiento debe reflejar lo que se sabía en el instante de la alerta. Cualquier reconstrucción documentará procedencia, tiempo de disponibilidad y confianza.

## Análisis de las dos secuencias con velocidad media cero

Se cotejaron los resúmenes de vuelta con las muestras de la caché `car_data`, por tiempo de vuelta. No se comprobó la causa en una fuente independiente.

- Japón 2022, BOT: las vueltas 3–28 (26 de 28 filas del piloto) tienen velocidad media 0, RPM media 0 y acelerador medio 104; en las vueltas 3 y 4 inspeccionadas, todas las 739 y 627 muestras originales respectivamente presentan velocidad 0, RPM 0 y acelerador 104. Las vueltas 1 y 2 muestran valores de velocidad y RPM diferentes de cero. La presencia de tiempos de vuelta y este patrón persistente son indicios de telemetría anómala; no permiten inferir la causa.
- Miami 2022, MAG: la vuelta 1 tiene velocidad y RPM medias 0 y acelerador medio 104; las 388 muestras originales dentro de su intervalo tienen esos mismos valores. En la vuelta 2 aparecen velocidad y RPM medias diferentes de cero. No se comprobó si otros canales o laps presentan anomalías diferentes.

Propuesta pendiente de acuerdo: preservar las filas y las muestras originales. Para los intervalos identificados con evidencia suficiente, representar los resúmenes de canales afectados como faltantes y acompañarlos de una marca de calidad, en una salida procesada; evaluar cada canal, sin asumir automáticamente que DRS o cualquier otro canal es inválido por su valor. No rellenar con vueltas futuras ni borrar carreras. Antes de aplicar una regla general, inspeccionar si hay patrones parciales o anomalías distintas y definir condiciones y pruebas; comparar modelos con una estrategia de imputación ajustada solo a entrenamiento y con indicadores de disponibilidad.

## Búsqueda adicional de patrones de telemetría

En 47.997 filas de 2022–2023 se observaron 27 vueltas con velocidad máxima cero, 28 con RPM máxima cero, 29 con acelerador medio superior a 100 y 427 con acelerador máximo superior a 100. Estos indicadores son **candidatos de revisión**, no pruebas independientes de invalidez. Un máximo superior a 100 por sí solo no establece la proporción de muestras afectadas.

También hay 73 vueltas con velocidad media inferior a 10, entre ellas una concentración en Mónaco 2022. Una velocidad baja puede corresponder a detenciones o interrupciones; no debe invalidarse automáticamente mediante ese umbral.

Se contrastaron ejemplos con las muestras originales dentro del intervalo registrado de vuelta: LEC, Azerbaiyán 2022 vuelta 21, tiene RPM cero en 503 muestras y velocidades entre 0 y 24; RUS, Australia 2023 vuelta 18, tiene 11.630 valores de acelerador 104 de 11.878 muestras; SAR, Países Bajos 2023 vuelta 15, tiene 14.623 valores 104 de 14.909 muestras. Los dos últimos registros son `FastF1Generated=True` y no tienen `LapTime`; contienen además otras velocidades o RPM. Estos ejemplos indican que un promedio o máximo no distingue muestras anómalas aisladas de series extensas o intervalos generados. Los conteos comparan resúmenes existentes y muestras de caché sin verificar la causa ni una referencia externa.

Propuesta para especificación: producir indicadores por canal e intervalo y recuentos de cobertura, documentando la regla, antes de anular resúmenes. Investigar el significado de 104 y la duración/condiciones de intervalos generados; no recortar, imputar ni invalidar carreras completas solo por un máximo o velocidad baja. Cualquier agregación recalculada debe usar muestras válidas disponibles hasta el instante de predicción y registrar cuántas se descartaron. Validar criterios con casos normales, pit stops, interrupciones y datos generados antes de implementarlos.

## Comparación de selección de muestras para acelerador

Se compararon siete vueltas de 2022 en Baréin (GAS), Miami (MAG) y Japón (BOT), cargando sesiones desde caché sin conexión. En vueltas con datos válidos hubo diferencias: GAS vuelta 1, 387 frente a 385 muestras válidas y media 59,421 frente a 59,605; MAG vuelta 2, 213 frente a 215 y media 57,704 frente a 58,088; BOT vuelta 2, 699 frente a 700 y media 23,757 frente a 23,723. En tres vueltas con todas las muestras de acelerador no válidas ambos métodos dieron cero muestras válidas; en GAS vuelta 45 coincidieron 61 muestras y la media.

La actualización local filtró el `Time` de las muestras crudas; FastF1 `Lap.get_car_data()` selecciona por `SessionTime` y reconstruye ese tiempo a partir de `Date` y `t0_date` de la sesión. La diferencia de referencia temporal explica por qué no se puede asumir equivalencia. Esos resultados son una prueba en siete vueltas, no una medición de diferencias globales. Las cuatro columnas de acelerador calculadas por la actualización local no deben considerarse definitivas ni compararse con carreras nuevas hasta unificar el método. La siguiente corrección debe recalcularlas por el mismo método de FastF1, usando caché local y verificando cobertura y errores, sin llamadas a la API cuando sea posible.

**Corrección posterior:** se recalcularon las cuatro columnas de acelerador de las 67 carreras guardadas usando `SessionTime` de FastF1, en modo sin conexión, y se regeneró el consolidado parcial. Los tres ejemplos que diferían (GAS 1, MAG 2 y BOT 2) coincidieron después en conteo y media con la selección de FastF1. Esto comprueba esos ejemplos, no constituye una comparación exhaustiva de todas las vueltas. El dataset aún es parcial y las etiquetas no están validadas.

## Clasificación de las filas `FastF1Generated`

Se contrastaron las 79 filas de 2022–2023 con vueltas de la sesión y estado de resultados cargados desde caché. Las 79 son la última vuelta registrada de su piloto. Todas tienen `LapTime`, `PitInTime`, `PitOutTime` y `Position` ausentes; 71 tienen estado final de retiro o incidente y 8 constan como finalizados. Esto confirma que la marca no equivale directamente a abandono: FastF1 también añade una fila final generada a pilotos clasificados.

Propuesta: conservar la fila para auditoría y estado terminal, pero no contarla como vuelta completa para la secuencia ni como evidencia de parada. La clasificación final puede señalar el cierre de la secuencia; no determina por sí sola si hubo una parada en un horizonte previo ni cuándo se supo que el piloto abandonó. Documentado también en `docs/specs/04_limpieza_candidata.md`.
