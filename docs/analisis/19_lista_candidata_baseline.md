# Lista candidata de variables para el baseline

**Estado:** borrador para socializar. Resume las decisiones conversadas y propone cerrar las candidatas restantes. No es todavía una configuración ejecutable ni una matriz aprobada para entrenar.

**Objetivo de la muestra:** al inicio de la vuelta N, predecir una visita confirmada por PitInTime/PitOutTime en N, N+1 o N+2. Ningún dato posterior a la decisión entra en X.

## Rama estática FFNN

| Variable | Estado | Regla |
|---|---|---|
| Driver | Acordada como parte del contexto estático | Categoría; vocabulario aprendido solo en train y categoría para desconocidos. |
| Team | Acordada como parte del contexto estático | Categoría; vocabulario de train y categoría para desconocidos. |
| GranPremio/circuito | Acordada como parte del contexto estático | Categoría; no confundir el nombre con características físicas del circuito. |

## Rama dinámica RNN

La lista reúne candidatas: cada campo necesita valor disponible en el instante de decisión y máscara de faltantes apropiada.

| Grupo/variable | Estado | Regla de entrada propuesta |
|---|---|---|
| LapNumber | Candidata del baseline | Progreso conocido al inicio de N; no derivar desde la última vuelta real de la carrera. |
| Compound_asof | Propuesta para socializar como candidata base | Último compuesto conocido por unión temporal hasta `decision_time`; categoría `UNKNOWN` si no se conoce o hay conflicto de evento. No usar la fila actual reconstruida. |
| TyreLife_last_eligible | Propuesta para socializar como candidata base | Usar `TyreLife` de la última vuelta cerrada con `Time <= decision_time`; máscara separada cuando falte o el valor aún no esté disponible. No inferirlo de otros campos. |
| Stint_asof | Pendiente de cierre | Estado/contador de stint observado as-of; validar numeración, empates de la fuente y los seis cambios sin pareja contigua. |
| FreshTyre_asof | Pendiente de cierre | Último `New` observado as-of; reconciliar diferencias con `FreshTyre` consolidado. No inferirlo de `TyreLife`. |
| LapTime, Sector1Time, Sector2Time, Sector3Time | Acordadas como variables de vueltas cerradas | Solo vueltas terminadas y disponibles. Las 858 reconstrucciones posibles de Sector1Time siguen sujetas a regla/elegibilidad. |
| SpeedI1, SpeedI2, SpeedST | Candidatas pendientes | Historial cerrado, revisar posible propagación (*forward fill*) de FastF1. |
| SpeedFL | Candidata solo histórica | Nunca incluir su valor ni su máscara en la vuelta actual: su ausencia casi identifica PitInTime de esa vuelta. |
| Position | Candidata acordada para el historial previo | Usar posición de la última vuelta completada si Time <= decision_time; revisar las 18 excepciones de reloj y empates. Nunca usar Position de la vuelta actual. |
| track_phase_asof | Acordada como derivación | Último cambio de pista anterior/simultáneo a la decisión; estado 7 distingue VSC terminando. Pausar alertas durante bandera roja. |
| Weather_AirTemp, Humidity, Pressure, Rainfall, TrackTemp, WindDirection, WindSpeed | Acordadas como candidatas derivadas | Unión hacia atrás por carrera, K_clima=120 s, latencia cero supuesta en primer replay y pruebas con retraso. Viento circular como seno/coseno. |
| Telemetry_Speed_mean/max | Acordadas como candidatas | Solo vueltas cerradas no generadas; valores cero y cobertura necesitan máscara/regla. Comparar media frente a máximo por ablación. |
| Telemetry_RPM_mean/max | Acordadas como candidatas | Solo vueltas cerradas no generadas; revisar ceros y extremos. Comparar media frente a máximo por ablación. |
| Telemetry_Throttle_valid_mean/max | Acordadas como candidatas | Solo vueltas cerradas; usar versiones 0–100, mantener NA sin muestras válidas. No incluir las versiones brutas. |
| Telemetry_Brake_mean | Acordada como candidata | Solo vuelta cerrada; se interpreta como proporción de muestras frenando, no duración. |

## Fuera de X o solo para auditoría

| Campo/familia | Estado y motivo |
|---|---|
| Temporada, Ronda, DriverNumber, Time, LapStartTime, LapStartDate | Claves/reloj para agrupar, ordenar y validar; no entran crudos como predictores. |
| PitInTime, PitOutTime, PitWithin3Laps y fuentes de etiqueta | Exclusivamente etiquetas/auditoría; nunca X. |
| SafetyCar, VirtualSafetyCar, TrackStatus de la fila actual | Excluir agregados retrospectivos; usar track_phase_asof. |
| Weather_Time | Timestamp/edad/auditoría; no magnitud meteorológica del modelo. |
| IsPersonalBest, Deleted, DeletedReason, FastF1Generated, IsAccurate | Auditoría/criterios de elegibilidad; no predictores automáticos. |
| Telemetry_Samples, Telemetry_Throttle_valid_samples/fraction | Cobertura/auditoría; no X en el baseline inicial. |
| Telemetry_Throttle_mean/max | Excluir: el código 104 no es una medida de acelerador 0–100. |
| Telemetry_Brake_max | Excluir: True en 47.960 de 47.997 filas; casi constante y redundante con Brake_mean. |
| Telemetry_nGear_mean/max | Excluir hasta recalcular desde muestras de marchas válidas 1–8. |
| Telemetry_DRS_mean/max | Excluir hasta contar con semántica validada; promediar códigos no es significativo. |
| Position de la vuelta actual | Excluir por disponibilidad posterior al inicio de esa vuelta. |

## Decisión acordada para velocidad, RPM y freno

En filas 2022–2023 no generadas, con LapTime presente, la correlación de Pearson media/máximo es 0,648 para velocidad, 0,297 para RPM y 0,101 para acelerador válido. Estas cifras descriptivas sugieren que medias y máximos no son intercambiables, sobre todo en RPM y acelerador; no demuestran que ambos ayuden al modelo. Se acordó incluir velocidad media/máxima, RPM media/máxima y Brake_mean como candidatas del primer baseline comparable, y medir su aporte con ablaciones posteriores. Los hiperparámetros y métricas no se seleccionarán usando prueba.

## Condiciones antes de convertirla en configuración ejecutable

1. Socializar y aprobar/rechazar las tres propuestas de telemetría de la tabla anterior.
2. Resolver la disponibilidad temporal de estado de neumáticos, SpeedI1/I2/ST y posición previa.
3. Reconciliar las 18 transiciones de reloj anómalas y las 858 posibles reconstrucciones de Sector1Time.
4. Regenerar las etiquetas con el contrato PitInTime/PitOutTime emparejados.
5. Generar un registro de características versionado con unidad, fuente, available_at, regla de faltantes, transformación y rama.
6. Solo después construir la matriz de características y ejecutar las comprobaciones contra fuga temporal.
