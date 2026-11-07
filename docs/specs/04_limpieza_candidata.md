# Propuesta de reglas de calidad y limpieza

**Estado:** borrador para revisión; no implementado. Basado en una exploración de las 44 carreras de 2022–2023 disponibles. No se han usado resultados de modelos para elegir reglas.

## Principios

Conservar una matriz maestra y archivos de fuente sin sobrescribir. La limpieza debe producir una versión diferenciada, motivos y recuentos de cambios. Las columnas necesarias para etiquetar y auditar permanecen en la matriz maestra, pero se excluyen explícitamente de las entradas del modelo. Las estadísticas para imputar o escalar se aprenden solo del conjunto de entrenamiento.

## Reglas candidatas

| Situación | Tratamiento propuesto | Validación pendiente |
| --- | --- | --- |
| Claves nulas o duplicadas | Marcar como error de integridad; no elegir una fila arbitrariamente. | Definir resolución y pruebas con casos reales. |
| Vueltas faltantes | Registrar discontinuidad por piloto y carrera; no inventar vueltas ni cruzar huecos sin regla. | Investigar comienzos, finales y datos omitidos con fuente independiente. |
| `PitInTime`/`PitOutTime` nulos | Preservar como ausencia de evento registrado; no imputar cero. | Falta de registro no prueba ausencia real del evento. |
| `TyreLife` u otras mediciones faltantes | Conservar faltantes y reportar por carrera y piloto; estudiar imputación y marcador de ausencia en cada experimento. | Confirmar si la información faltante es recuperable y está disponible en el momento de predicción. |
| `Compound` igual a texto `None` | Tratar como categoría especial a investigar; no confundir cadena con nulo real. | Confirmar semántica y decisión de recodificación. |
| `FastF1Generated=True` | Mantener la bandera; analizar cada uso y no tratar inferencias como mediciones observadas. | Definir reglas para etiquetas y ventanas. |
| `IsAccurate=False` | No eliminar automáticamente; conservar la bandera para auditoría y revisar la calidad de cada característica. | Confirmar si el indicador y sus componentes están disponibles en el instante de predicción antes de incluirlos en un modelo. |
| Telemetría anómala o sin cobertura | Marcar la medición como potencialmente inválida, no reemplazarla por cero. Mantener datos originales. | Definir criterios y umbrales basados en las series originales y contrastarlos. |
| Clima y telemetría por vuelta completa | Al iniciar la vuelta, usar solo observaciones o resúmenes ya disponibles. Para clima, buscar la última observación anterior o simultánea al instante. | Verificar latencias, reloj, estado inicial y disponibilidad de datos originales. |

## Casos investigados

En 2022–2023 no se observaron saltos mayores a una vuelta entre filas consecutivas de un mismo piloto y carrera; esto no prueba cobertura completa al inicio o final ni ausencia de vueltas artificiales. Los 555 `TyreLife` faltantes incluyen 303 filas con la cadena `None` como compuesto.

En Japón 2022, BOT tiene 26 vueltas con velocidad media cero; se inspeccionó la caché original del piloto y se observaron largos tramos de velocidad y RPM en cero y acelerador 104. Son anomalías candidatas de la fuente, no prueba del motivo. La muestra también tiene una vuelta de MAG en Miami 2022 con velocidad media cero; no se ha investigado su causa. No fijar un umbral general ni modificar la telemetría sin validación adicional.

## Criterios de aceptación antes de implementación

- Las reglas se aplican por separado a copias o salidas nuevas y dejan trazabilidad de cada cambio.
- Se verifican claves, cobertura y nulos con un informe por carrera y piloto.
- Ejemplos representativos comprueban que una vuelta de entrada no se elimina solo por `SpeedFL` nulo o `IsAccurate=False`.
- Pruebas temporales confirman que ninguna observación posterior al instante de predicción entra a la ventana.
- El tratamiento de datos anómalos, ausencias y vueltas generadas se aprueba tras revisar casos y medir su impacto.
- Las etiquetas definitivas se generan únicamente tras validar la regla específica de cambio de neumáticos.

## Caso que requiere decisión: Austria 2022

Los compuestos desconocidos se concentran en Austria 2022: 299 de 1.324 filas afectan a 12 pilotos. La aclaración y los costos de excluir la carrera constan en `docs/aclaraciones/01_compuestos_desconocidos.md`. Conservar la fuente y la matriz maestra; la exclusión para experimentos es una opción pendiente de comparar y no una regla aprobada. Una decisión de tratamiento debe registrar su impacto y no usar información posterior al instante de predicción.

## Vueltas generadas por FastF1

Exploración de los archivos 2022–2023 disponibles: 79 filas marcadas `FastF1Generated`, todas últimas filas registradas para su piloto en la carrera. Las 79 carecen de `LapTime`, `PitInTime`, `PitOutTime` y `Position`. El estado de resultados asociado clasifica 71 como retiro o incidente y 8 como finalizados. Por tanto, `FastF1Generated=True` no equivale automáticamente a abandono.

Propuesta: conservar las filas originales para auditoría y contexto del cierre, pero no tratarlas como vueltas completas para formar ventanas de entrenamiento ni como evidencia de parada. Usar estado final y datos de sesión para determinar el final de la secuencia del piloto; una fila generada por sí sola no basta para confirmar abandono ni para etiquetar ausencia de parada. La definición de horizonte al final de una carrera y el momento en que un abandono se confirma siguen pendientes de regla explícita. Esta propuesta debe mantenerse separada de la etiqueta booleana; si la muestra no tiene etiqueta verificable, marcarla desconocida.
