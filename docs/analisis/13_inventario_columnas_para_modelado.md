# Inventario de columnas para decidir variables del modelo

**Fuente inspeccionada:** `EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_limpieza_v1.pkl`.

**Corte de los recuentos:** temporadas 2022–2023, 47.997 filas, 44 carreras y 61 columnas. El archivo completo tiene 101.290 filas. Los recuentos de faltantes son `pandas.isna()` y no detectan valores físicamente inválidos ni información recibida tarde.

**Estado:** inventario para socializar, no lista final de entradas ni matriz aprobada. Se aplica el [contrato temporal](../specs/08_dataset_causal_modelado.md): predecir al iniciar la vuelta `N` con información conocida hasta ese instante. La rama FFNN recibe contexto estático; la RNN, historial y estado dinámico disponibles. La [etiqueta revisada](../specs/07_objetivo_entrada_boxes.md) exige `PitInTime` y `PitOutTime` emparejados.

## Cómo leer las decisiones provisionales

- **Estática FFNN:** candidata de la rama estática; codificación y vocabulario ajustados solo en train.
- **Dinámica RNN:** candidata de la rama temporal únicamente después de aplicar la regla indicada. El valor de la fila de la vuelta que empieza no se entrega completo al modelo.
- **Clave/reloj:** sirve para identificar, ordenar, partir o reconstruir el instante de decisión; puede originar una característica derivada, pero no se introduce directamente como valor sin aprobarla.
- **Auditoría:** se conserva fuera de `X` para calidad o procedencia. **Etiqueta/auditoría:** además puede determinar `y`, pero nunca entra en `X`.
- **Excluir por ahora:** no entra en el experimento base; requiere una derivación nueva o evidencia adicional para reconsiderarse.

Los faltantes de `PitInTime` y `PitOutTime` son normales para eventos escasos. Los valores ausentes de predictores se mantienen como tales hasta el preprocesamiento del experimento; no se convierten en cero en la matriz canónica.

## Identidad, reloj y progreso

| # | Columna | Tipo | NA | Papel propuesto | Decisión que debemos revisar |
|---:|---|---|---:|---|---|
| 1 | `Temporada` | entero | 0 | Clave/reloj | Identifica temporada y partición cronológica. Excluir del `X` base para no codificar directamente el año de train/validación/prueba. |
| 2 | `Ronda` | entero | 0 | Clave/reloj | Identifica carrera dentro de la temporada; no interpretar el número como dificultad o distancia. Puede originar progreso del campeonato en otro experimento. |
| 3 | `GranPremio` | texto | 0 | Estática FFNN | Categoría de circuito/evento conocida antes de la salida; verificar nombres equivalentes entre años. No equivale a características físicas de pista. |
| 4 | `Time` | duración | 0 | Clave/reloj | Tiempo de referencia de la fila al cierre; usar para reconstrucción temporal, no como predictor de la vuelta que comienza. |
| 5 | `Driver` | texto | 0 | Estática FFNN | Identidad del piloto. Tratar como categoría; medir ablación sin identidad para detectar memorización. |
| 6 | `DriverNumber` | texto | 0 | Clave/reloj | Clave estable del piloto dentro de la carrera. Evitar duplicarlo con `Driver` o tratarlo como magnitud numérica. |
| 8 | `LapNumber` | decimal | 0 | Dinámica RNN | Número/progreso de la vuelta actual conocido al cruzar la línea. Validar entero positivo y no confundirlo con la cantidad de filas observadas. |
| 26 | `Team` | texto | 0 | Estática FFNN | Categoría del equipo conocido en la carrera; comprobar que la inscripción usada precede a la decisión. |
| 27 | `LapStartTime` | duración | 0 | Clave/reloj | Ancla propuesta de `decision_time`; derivar tiempo transcurrido si se justifica. No alimentar como tiempo absoluto crudo. |
| 28 | `LapStartDate` | fecha/hora | 0 | Clave/reloj | Referencia de calendario/UTC para alinear fuentes. Excluir como fecha absoluta cruda; revisar coherencia con el reloj de sesión. |

## Neumáticos y eventos de boxes

| # | Columna | Tipo | NA | Papel propuesto | Decisión que debemos revisar |
|---:|---|---|---:|---|---|
| 9 | `Stint` | decimal | 0 | Dinámica RNN | Estado del stint solo si el número ya era observable. La asignación histórica de la fila actual puede conocerse después de la parada. |
| 10 | `PitOutTime` | duración | 46.233 | Etiqueta/auditoría | Emparejar con `PitInTime`; nunca entra en `X`. Una salida sola no es parada positiva bajo el contrato nuevo. |
| 11 | `PitInTime` | duración | 46.218 | Etiqueta/auditoría | Ancla de la visita si existe salida válida emparejada; nunca entra en `X`. Una entrada sola queda incierta. |
| 23 | `Compound` | texto | 36 | Dinámica RNN | Último compuesto conocido antes de decidir. Una reconstrucción posterior no puede rellenar retrospectivamente `X` sin prueba de disponibilidad. |
| 24 | `TyreLife` | decimal | 555 | Dinámica RNN | Edad observada del neumático; distinguirla de una edad inferida y verificar su instante de disponibilidad. |
| 25 | `FreshTyre` | booleano | 0 | Dinámica RNN | Estado del juego solo después de conocerlo; no asumir que el valor de la fila inicial del stint estaba disponible antes del pit out. |

## Ritmo, sectores y posición

| # | Columna | Tipo | NA | Papel propuesto | Decisión que debemos revisar |
|---:|---|---|---:|---|---|
| 7 | `LapTime` | duración | 270 | Dinámica RNN | Tiempo de vuelta completada y disponible; nunca usar el tiempo final de la vuelta que comienza. |
| 12 | `Sector1Time` | duración | 1.041 | Dinámica RNN | Tiempo sector 1 de vueltas cerradas; revisar faltantes y disponibilidad. |
| 13 | `Sector2Time` | duración | 89 | Dinámica RNN | Tiempo sector 2 de vueltas cerradas; revisar faltantes y disponibilidad. |
| 14 | `Sector3Time` | duración | 178 | Dinámica RNN | Tiempo sector 3 de vueltas cerradas; revisar faltantes y disponibilidad. |
| 15 | `Sector1SessionTime` | duración | 1.147 | Clave/reloj | Marca acumulada de paso por sector; útil para ordenar/validar, no como señal adicional de ritmo sin evaluar redundancia. |
| 16 | `Sector2SessionTime` | duración | 89 | Clave/reloj | Igual: tiempo de sesión para alineación; no usar el paso de un sector futuro de la vuelta actual. |
| 17 | `Sector3SessionTime` | duración | 178 | Clave/reloj | Igual: ayuda a cerrar la vuelta; comprobar coherencia temporal con `Time`. |
| 18 | `SpeedI1` | decimal | 80 | Dinámica RNN | Velocidad en punto de control de vuelta cerrada y disponible; no es velocidad instantánea al decidir. |
| 19 | `SpeedI2` | decimal | 79 | Dinámica RNN | Mismo tratamiento para segundo punto de control. |
| 20 | `SpeedFL` | decimal | 1.857 | Dinámica RNN | Velocidad de línea final; faltantes relativamente frecuentes, no imputar con cero. |
| 21 | `SpeedST` | decimal | 113 | Dinámica RNN | Velocidad en speed trap de vuelta cerrada y disponible. |
| 22 | `IsPersonalBest` | booleano almacenado como texto/objeto | 64 | Excluir por ahora | Indicador retrospectivo de la vuelta. Un histórico causal podría derivarse después, pero el base usará tiempos observados. |
| 30 | `Position` | decimal | 79 | Dinámica RNN | Última posición con timestamp verificable; no usar la clasificación final ni asumir que la fila actual representa el inicio. |

## Banderas de calidad y revisión

| # | Columna | Tipo | NA | Papel propuesto | Decisión que debemos revisar |
|---:|---|---|---:|---|---|
| 31 | `Deleted` | booleano | 0 | Auditoría | Indica revisión/eliminación retrospectiva de la vuelta; no implica que sus mediciones físicas sean inválidas ni estaba necesariamente disponible al decidir. |
| 32 | `DeletedReason` | texto | 64 | Auditoría | Motivo retrospectivo de la revisión; excluir de `X` y conservar para diagnóstico. |
| 33 | `FastF1Generated` | booleano | 0 | Auditoría | Marca fila generada por la biblioteca; sirve para reglas de elegibilidad y cobertura de etiquetas, no como predictor de estrategia. |
| 34 | `IsAccurate` | booleano | 0 | Auditoría | Evaluación retrospectiva de exactitud de la vuelta; no usar como predictor ni descartar todas las filas `False` sin análisis. |

## Estado de pista y clima

| # | Columna | Tipo | NA | Papel propuesto | Decisión que debemos revisar |
|---:|---|---|---:|---|---|
| 29 | `TrackStatus` | texto | 0 | Excluir por ahora | Contiene códigos observados durante la vuelta completa. Derivar estado causal desde mensajes con timestamp, no dar el texto actual directamente. |
| 35 | `SafetyCar` | booleano nullable | 0 | Excluir por ahora | Bandera retrospectiva de cualquier código SC en la vuelta. Sustituir por `safety_car_asof` calculado al decidir. |
| 36 | `VirtualSafetyCar` | booleano nullable | 0 | Excluir por ahora | Misma fuga potencial; sustituir por `virtual_safety_car_asof`. |
| 37 | `Weather_Time` | duración | 0 | Clave/reloj | Timestamp de observación; seleccionar hacia atrás y derivar edad. La observación asignada a la fila suele ocurrir después de `LapStartTime`. |
| 38 | `Weather_AirTemp` | decimal | 0 | Dinámica RNN | Temperatura del aire de la última observación disponible dentro de `K_clima`; no usar directamente la de la fila actual. |
| 39 | `Weather_Humidity` | decimal | 0 | Dinámica RNN | Humedad con la misma unión temporal y límite de antigüedad. |
| 40 | `Weather_Pressure` | decimal | 0 | Dinámica RNN | Presión con la misma unión temporal y unidades documentadas. |
| 41 | `Weather_Rainfall` | booleano | 0 | Dinámica RNN | Lluvia observada hasta la decisión; no usar la bandera de una observación posterior. |
| 42 | `Weather_TrackTemp` | decimal | 0 | Dinámica RNN | Temperatura de pista con unión temporal y `K_clima`. |
| 43 | `Weather_WindDirection` | entero | 0 | Dinámica RNN | Ángulo circular; codificar con seno/coseno solo sobre la observación temporalmente válida. |
| 44 | `Weather_WindSpeed` | decimal | 0 | Dinámica RNN | Velocidad del viento con unión temporal y control de unidades. |

## Telemetría agregada por vuelta

Todos estos agregados describen la **vuelta completa**. Pueden entrar en la RNN solo como historial cuyo cierre y disponibilidad preceden a la decisión. El dataset actual no contiene aquí una secuencia cruda por instante. Los tipos y faltantes siguientes corresponden a los agregados existentes, no a una telemetría ya validada.

| # | Columna | Tipo | NA | Papel propuesto | Decisión que debemos revisar |
|---:|---|---|---:|---|---|
| 45 | `Telemetry_Samples` | entero | 0 | Auditoría | Cobertura/calidad del agregado; hay conteos extremos. No usar como predictor base sin verificar qué mide en cada caso. |
| 46 | `Telemetry_Speed_mean` | decimal | 0 | Dinámica RNN | Velocidad media tras revisar ceros y cobertura; puede reflejar neutralización o problemas de canal. |
| 47 | `Telemetry_Speed_max` | decimal | 0 | Dinámica RNN | Máximo de velocidad de vuelta cerrada; validar rango y cobertura. |
| 48 | `Telemetry_RPM_mean` | decimal | 0 | Dinámica RNN | RPM media tras revisar valores cero y muestras inválidas. |
| 49 | `Telemetry_RPM_max` | decimal | 0 | Dinámica RNN | Máximo de RPM tras validar rango/canal. |
| 50 | `Telemetry_Throttle_mean` | decimal | 0 | Excluir por ahora | Agregado bruto contiene valores inválidos >100; preferir versión validada, sin duplicar señales. |
| 51 | `Telemetry_Throttle_max` | decimal | 0 | Excluir por ahora | Incluye código 104 de acelerador no disponible; no tratarlo como porcentaje real. |
| 52 | `Telemetry_Brake_mean` | decimal | 0 | Dinámica RNN | Fracción/media de frenado solo tras confirmar codificación y cobertura de muestras. |
| 53 | `Telemetry_Brake_max` | booleano | 0 | Excluir por ahora | Indica si hubo al menos una muestra con freno activado; revisar si aporta algo sobre `Brake_mean`. |
| 54 | `Telemetry_nGear_mean` | decimal | 0 | Dinámica RNN condicionada | Algunos promedios superan 8; recalcular con muestras de marcha válidas antes de habilitar. |
| 55 | `Telemetry_nGear_max` | entero | 0 | Dinámica RNN condicionada | Máximos llegan a 128; no introducir ni recortar silenciosamente. Validar y recalcular. |
| 56 | `Telemetry_DRS_mean` | decimal | 0 | Excluir por ahora | Media de códigos discretos DRS sin significado físico directo; definir primero una derivación semántica. |
| 57 | `Telemetry_DRS_max` | entero | 0 | Excluir por ahora | Máximo del código, no magnitud de apertura; requiere diccionario y derivación. |
| 58 | `Telemetry_Throttle_valid_samples` | entero | 0 | Auditoría | Cobertura de muestras válidas de acelerador; candidato eventual como indicador de calidad, no señal física directa. |
| 59 | `Telemetry_Throttle_valid_fraction` | decimal | 0 | Auditoría | Proporción válida del canal; útil para enmascarar/describir calidad. Revisar antes de hacerla predictor. |
| 60 | `Telemetry_Throttle_valid_mean` | decimal | 27 | Dinámica RNN | Media de acelerador filtrada 0–100; usar solo con cobertura suficiente y vuelta ya disponible. |
| 61 | `Telemetry_Throttle_valid_max` | decimal | 27 | Dinámica RNN | Máximo filtrado; revisar aporte frente a media y cobertura. |

## Columnas externas y pasos de revisión

`PitWithin3Laps` está en el artefacto de etiquetas, **no** entre estas 61 columnas. Solo es `y`; las etiquetas v19 aún no siguen la regla nueva de pareja obligatoria y deberán regenerarse antes de entrenar. `safety_car_asof`, `virtual_safety_car_asof` y `weather_age_seconds` son características propuestas, no columnas actuales.

Para decidir una columna se registrará: **usar/excluir/derivar**, instante de disponibilidad, regla de limpieza, manejo de faltantes, rama FFNN/RNN y prueba de que no cambia al alterar el futuro. Aprobar una familia no aprueba automáticamente cada uno de sus campos. El orden sugerido de discusión es: (1) identidad/progreso, (2) neumáticos, (3) ritmo/sectores/posición, (4) estado de pista/clima, (5) telemetría. La revisión de cada familia actualizará el registro de características del experimento base.
