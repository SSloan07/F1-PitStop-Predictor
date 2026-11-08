# Revisión 4: estado de pista y clima al instante de alerta

**Fuente local:** matriz `vueltas_2022_2025_limpieza_v1.pkl` y cachés `track_status_data.ff1pkl`/`weather_data.ff1pkl` de las 44 carreras 2022–2023. Se revisaron 47.997 decisiones piloto-vuelta. No se hicieron llamadas a la API ni se modificaron datos. Esta revisión desarrolla las 11 columnas del [inventario](13_inventario_columnas_para_modelado.md) correspondientes a pista y clima.

**Estado:** propuestas para socializar, no características construidas ni aprobadas para entrenamiento. El modelo decide al iniciar una vuelta; el dato debe haberse producido y estar disponible en ese instante. Los `Time` de las cachés son marcas del evento y **no** prueban cuándo llegó el mensaje a un consumidor en vivo.

## Estado de pista: la fila completa mira más allá de la decisión

El extractor actual marca `SafetyCar` si el texto `TrackStatus` de **toda la vuelta** contiene el código `4`, y `VirtualSafetyCar` si contiene `6` o `7`. `TrackStatus` tiene 39 valores distintos y **4.297 filas con más de un código**. Hay **133 filas** en las que ambas banderas son verdaderas, lo que indica presencia de ambos estados en algún momento de la vuelta, no simultaneidad al inicio. La [documentación de FastF1](https://raw.githubusercontent.com/theOehrly/Fast-F1/main/fastf1/_api.py) define `1` pista libre, `2` amarilla, `4` Safety Car, `5` bandera roja, `6` VSC y `7` VSC terminando; `1` marca el final efectivo de VSC.

Las 44 cachés de `track_status_data` contienen **569 cambios de estado**, entre 1 y 31 por carrera. Buscando el último cambio con `Time <= LapStartTime`, existe un estado previo en todas las 47.997 filas del corte. La distribución de este **replay con latencia cero supuesta** es: `1` libre 44.855, `2` amarilla 525, `4` Safety Car 2.036, `5` roja 14, `6` VSC 512 y `7` VSC terminando 55.

La diferencia frente a las banderas por vuelta revela el riesgo de fuga:

| Bandera retrospectiva de la fila | Filas `True` | De ellas, estado vigente al inicio distinto de esa bandera |
|---|---:|---:|
| `SafetyCar` | 2.769 | **733** |
| `VirtualSafetyCar` | 1.163 | **596** |

Estas diferencias pueden ser cambios ocurridos **después** del inicio de la vuelta; no son necesariamente errores del feed. `SafetyCar`, `VirtualSafetyCar` y `TrackStatus` de la fila que empieza **no entrarán directamente en `X`**. Se propone derivar `track_phase_asof` del último evento de pista disponible hasta `decision_time`. El código `7` conservará su estado propio `VSC_ENDING`; una bandera derivada `virtual_safety_car_asof` puede mantenerse verdadera para `6` y `7`, y `safety_car_asof` solo para `4`. La categoría completa evita confundir una amarilla, una roja y una pista libre. Las banderas derivadas y la categoría son representaciones alternativas o ablaciones, no tres señales independientes por accidente.

La bandera roja merece una regla de elegibilidad aparte: el estado `5` aparece en 14 decisiones candidatas. Se propone **no emitir alerta durante una suspensión roja**, guardar el motivo y reanudar solo cuando la sesión/estado vuelva a permitir carrera. La interacción con [las transiciones de pit lane y reloj](14_revision_identidad_reloj_progreso.md) debe revisarse antes de cerrar esa propuesta.

## Clima: observación anterior y tolerancia configurable

El extractor usa `laps.get_weather_data()`. Según la [implementación de FastF1](https://raw.githubusercontent.com/theOehrly/Fast-F1/main/fastf1/core.py), esa función elige la primera observación **dentro** de la vuelta o, si no la hay, la última antes de terminarla. Por ello, `Weather_Time` de la fila actual es posterior a `LapStartTime` en **47.973 de 47.997 filas**; hay una igualdad y 23 observaciones anteriores. Los ocho campos meteorológicos sin `NA` en la matriz no son, por eso, ocho características listas para decidir al inicio.

Las 44 cachés crudas contienen **7.425 observaciones** meteorológicas, 108–255 por carrera, sin timestamps duplicados dentro de carrera. Haciendo una unión temporal **hacia atrás** por carrera con el último `Time <= LapStartTime`, las 47.997 decisiones encuentran una observación anterior o simultánea. Su edad tiene mediana **29,99 s**, percentil 90 **54,18 s**, percentil 99 **59,44 s** y máximo **99,105 s**. Cobertura según la antigüedad máxima:

| `K_clima` | Decisiones con observación anterior dentro del límite | Ausentes por expiración |
|---:|---:|---:|
| 60 s | 47.977 | 20 |
| 90 s | 47.996 | 1 |
| **120 s (propuesta base)** | **47.997** | **0** |

Se propone `K_clima=120 s` como **configuración inicial**, escogida solo con el corte 2022–2023; continuará siendo parámetro de experimento. El paquete registrará `weather_event_time`, `weather_age_seconds`, regla de latencia y máscara de expiración. En un replay con retraso de fuente distinto de cero, la cobertura puede bajar y deberá medirse de nuevo; no se cambiará el límite observando resultados de validación/prueba. Aunque la observación anterior exista, su disponibilidad real en vivo requiere un timestamp de recepción o un supuesto explícito de latencia.

La [referencia de FastF1](https://raw.githubusercontent.com/theOehrly/Fast-F1/main/fastf1/_api.py) indica °C para temperaturas, % para humedad, mbar para presión, grados para dirección del viento y m/s para velocidad del viento. En las 7.425 observaciones revisadas no hay valores nulos ni cero para aire, humedad, presión o pista; hay 251 direcciones de viento `0°` y 64 velocidades `0 m/s`, ambos valores posibles. El parser de FastF1 puede poner cero cuando una clave de fuente falta o no se puede convertir; por ello, una versión futura debe mantener validaciones físicas por campo y no confundir «sin nulos pandas» con calidad garantizada.

## Decisión propuesta por columna

| Columna original | Tratamiento en el experimento base | Regla temporal/de calidad |
|---|---|---|
| `TrackStatus` | Auditoría; derivar `track_phase_asof` para RNN. | Último cambio de estado de la carrera disponible al decidir, con código y tiempo de evento. |
| `SafetyCar` | Excluir bandera original; derivar `safety_car_asof`. | Verdadera únicamente si el estado vigente conocido es `4`. |
| `VirtualSafetyCar` | Excluir bandera original; derivar `virtual_safety_car_asof`. | Verdadera para `6` o `7` hasta que llegue cambio de estado. |
| `Weather_Time` | Ancla/auditoría; derivar edad y expiración. | Último registro anterior/simultáneo de la misma carrera, con latencia declarada. |
| `Weather_AirTemp` | Candidata RNN. | °C del registro seleccionado; `NA` si expira o es inválido. |
| `Weather_Humidity` | Candidata RNN. | % del mismo registro; validar rango. |
| `Weather_Pressure` | Candidata RNN. | mbar del mismo registro; cero no es una presión válida. |
| `Weather_Rainfall` | Candidata RNN. | Booleano del mismo registro; conservar `NA` cuando no hay observación válida. |
| `Weather_TrackTemp` | Candidata RNN. | °C del mismo registro; validar rango. |
| `Weather_WindDirection` | Candidata RNN derivada. | Codificar `sin`/`cos` de grados tras seleccionar el registro; `0°` es válido. |
| `Weather_WindSpeed` | Candidata RNN. | m/s del mismo registro; `0` puede ser calma real. |

## Comprobaciones para la transformación futura

1. Alterar un estado de pista o clima posterior a `decision_time` no cambia las características de esa decisión.
2. Reproducir una carrera prefijo a prefijo produce el mismo estado que el procesamiento batch con eventos filtrados temporalmente.
3. En una decisión donde SC/VSC inicia después del cruce, el estado actual permanece el anterior, aunque la bandera retrospectiva de la fila sea verdadera.
4. La búsqueda de clima nunca toma una observación futura, nunca cruza otra carrera y devuelve `NA` cuando la observación anterior supera `K_clima`.
5. Estado de pista desconocido, bandera roja, marca meteorológica ausente y valor meteorológico inválido tienen motivos distintos en auditoría; ninguno se rellena con `False` o cero por defecto.

**Puntos a cerrar juntos:** aceptar `K_clima=120 s` como base; decidir la suspensión de alertas durante bandera roja; declarar el supuesto inicial de latencia para la reproducción histórica. Las reglas de disponibilidad en vivo necesitarán medición real cuando exista el front/replay conectado.
