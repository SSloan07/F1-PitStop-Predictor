# Revisión 1: identidad, reloj y progreso de vuelta

**Fuente:** `EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_limpieza_v1.pkl`.

**Alcance del control principal:** carreras 2022–2023, 47.997 filas y 874 secuencias piloto-carrera. Se compararon categorías con 2024 y 2025 solo para anticipar valores desconocidos; no se usaron etiquetas ni métricas de modelo de esos años. Esta revisión desarrolla el primer grupo del [inventario de columnas](13_inventario_columnas_para_modelado.md) y aplica el [contrato temporal](../specs/08_dataset_causal_modelado.md). No modifica la matriz ni aprueba su uso para entrenar.

## Controles realizados

| Control | Resultado | Lectura |
|---|---:|---|
| Clave `(Temporada, Ronda, DriverNumber, LapNumber)` duplicada | 0 | Identifica unívocamente las filas inspeccionadas. |
| Faltantes en claves, `LapStartTime` o `LapStartDate` | 0 | Hay valores presentes; aún debe validarse su significado en casos excepcionales. |
| `LapNumber` fraccional, no positivo o fuera de 1–78 | 0 | Almacenar como entero en la versión canónica. |
| Secuencias piloto-carrera que no empiezan en vuelta 1 | 0 de 874 | La regla de ventana adaptativa puede comenzar con una sola posición en este corte. |
| Huecos o retrocesos internos de `LapNumber` | 0 | La numeración es continua en este corte. |
| `GranPremio`, `Driver` o `Team` que cambia dentro de una secuencia piloto-carrera | 0 | Funcionan como contexto estático en la carrera. |
| `GranPremio` que cambia dentro de una carrera | 0 | El nombre es estable en cada evento. |
| `LapStartTime` no creciente dentro de una secuencia | 0 | El orden general del reloj es coherente. |
| Diferentes valores de `LapStartDate − LapStartTime` dentro de una carrera | 0 de 44 | Los dos relojes comparten una referencia constante por carrera; esto no demuestra por sí solo la zona horaria ni la hora de recepción. |

En todas las carreras, los pilotos tienen el mismo `LapStartTime` registrado para la vuelta 1. Esto sirve como ancla de inicio de carrera en esta fuente; no demuestra que cada piloto cruzó físicamente la línea en el mismo instante. La definición de la primera muestra conservará su tratamiento especial de estado inicial conocido.

## Categorías nuevas fuera de train

La propuesta de partición usa 2022–2023 para entrenar, 2024 para validar y 2025 para prueba. Frente al vocabulario de train:

| Temporada | Pilotos `Driver` nuevos | Equipos `Team` nuevos | Nombres `GranPremio` nuevos |
|---|---:|---:|---:|
| 2024 | 3: `BEA`, `COL`, `DOO` | 2: `Kick Sauber`, `RB` | 1: `Chinese Grand Prix` |
| 2025 | 6: `ANT`, `BEA`, `BOR`, `COL`, `DOO`, `HAD` | 2: `Kick Sauber`, `Racing Bulls` | 1: `Chinese Grand Prix` |

`Driver` y `DriverNumber` no son intercambiables globalmente: `DEV` aparece con los números 45 en 2022 y 21 en 2023; `BEA` aparece con dos números en 2024. El número sí permanece válido como parte de la clave **dentro de cada carrera** inspeccionada. Los nombres de equipo pueden cambiar por temporada; no unificaremos marcas o sucesores por intuición. La primera codificación mantendrá las categorías tal como están y reservará un valor de desconocido para cada vocabulario aprendido solo con train. Una normalización de nombres posterior requerirá mapeo explícito y versión nueva.

## Excepciones del reloj entre vueltas

En 47.025 de 47.123 transiciones consecutivas de un piloto, `Time` de la vuelta anterior coincide exactamente con `LapStartTime` de la siguiente. Las **98 excepciones** aparecen en **seis carreras** y **todas** tienen `PitInTime` en la vuelta anterior y `PitOutTime` en la siguiente. Hay 80 diferencias positivas (pausas, hasta 7.578,858 segundos) y 18 negativas (solapamientos, hasta 47,084 segundos). Las 18 negativas se concentran en Australia 2023. Las seis carreras son Gran Bretaña y Japón 2022; Australia, Países Bajos, Ciudad de México y São Paulo 2023.

La inspección de `session_status_data` de FastF1 (`EDA/dataset/cache_fastf1_carreras/2023/2023-04-02_Australian_Grand_Prix/2023-04-02_Race/session_status_data.ff1pkl`) identifica las 18 diferencias negativas como vueltas de reinicio en Australia 2023: ocho filas comienzan en la vuelta 9 y diez en la 56. En ambos casos, el `LapStartTime` compartido coincide exactamente con el cambio de estado de sesión de `Aborted` a `Started` (`01:31:58.608` y `02:54:58.598`); el `Time` de la vuelta anterior queda registrado después de ese ancla. Esto explica la agrupación como límite de sesión/reinicio, pero no demuestra que el resumen de la vuelta anterior estuviera disponible para el modelo en el instante del reinicio.

No corregiremos marcas tomando el máximo/mínimo, interpolando ni moviendo decisiones a otra vuelta. Para las 18 muestras afectadas, la recomendación es conservar las filas fuente para trazabilidad, pero marcarlas como no elegibles para evaluación predictiva hasta fijar y verificar un ancla causal de reinicio. No se usarán como entrada mediciones de una vuelta cuyo cierre registrado es posterior a `decision_time`. Las otras 80 excepciones positivas siguen requiriendo auditoría propia; no se extrapola a ellas la explicación de los reinicios.

## Decisión propuesta por columna

| Columna | Decisión para el experimento base | Condición pendiente |
|---|---|---|
| `Temporada` | Clave/partición; fuera de `X`. | Mantener en manifiesto y evaluación por año. |
| `Ronda` | Clave de carrera; fuera de `X`. | No codificar ordinalmente una estrategia por número de ronda. |
| `GranPremio` | Categoría estática de la FFNN. | Vocabulario train con desconocido; nombres exactos y procedencia trazables. |
| `Time` | Marca de cierre para alineación; fuera de `X` crudo. | Aplicar elegibilidad causal a las 18 muestras de reinicio y revisar por separado las 80 pausas positivas. |
| `Driver` | Categoría estática de la FFNN. | Vocabulario train con desconocido y ablación sin identidad. |
| `DriverNumber` | Parte de la clave; fuera de `X` crudo. | No usar como identidad global ni número continuo. |
| `LapNumber` | Progreso dinámico conocido al inicio de la vuelta; candidato RNN. | Convertir a entero; no derivar progreso con la última vuelta real, que puede conocerse solo al final. |
| `Team` | Categoría estática de la FFNN. | Vocabulario train con desconocido; no fusionar nombres sin evidencia/mapeo. |
| `LapStartTime` | Ancla candidata de `decision_time`; fuera de `X` crudo. | Validar transiciones de pit lane y relación con el cruce físico. |
| `LapStartDate` | Alineación de fuentes/calendario; fuera de `X` crudo. | Documentar zona horaria y relación con `LapStartTime` antes de uniones temporales. |

## Comprobaciones para aceptar la transformación futura

1. Claves únicas, secuencias continuas y contexto estático por piloto-carrera en las temporadas que se publiquen.
2. Categorías nuevas transformadas por una política aprendida en train, sin ajustar vocabulario con validación/prueba.
3. Una muestra al inicio de `N` no contiene `Time`, sectores ni telemetría completos de `N`.
4. Las 98 transiciones excepcionales del corte actual tienen una decisión trazable: instante validado o muestra excluida con motivo. Las 18 de reinicio se mantienen fuera de evaluación hasta verificar el ancla causal; las 80 pausas positivas se resuelven por separado. Ninguna se corrige silenciosamente.
5. La muestra de vuelta 1 usa únicamente estado inicial disponible; no presupone una vuelta anterior ni emplea el resultado de la primera vuelta.

**Siguiente grupo para socializar:** `Stint`, `Compound`, `TyreLife` y `FreshTyre`. Su valor histórico puede ser correcto sin haber estado disponible al inicio de la vuelta, por lo que requiere una revisión temporal específica.
