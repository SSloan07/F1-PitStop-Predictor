# Propuesta de reglas de calidad y limpieza

**Estado:** primera regla de normalización histórica de `Compound` acordada e implementada; las demás reglas semánticas continúan en revisión. La auditoría conservadora permanece disponible y no modifica la matriz. Basado inicialmente en una exploración de las 44 carreras de 2022–2023; el perfil ampliado abarca la matriz consolidada 2022–2025. No se han usado resultados de modelos para elegir reglas.

## Principios

Conservar una matriz maestra y archivos de fuente sin sobrescribir. La limpieza debe producir una versión diferenciada, motivos y recuentos de cambios. Las columnas necesarias para etiquetar y auditar permanecen en la matriz maestra, pero se excluyen explícitamente de las entradas del modelo. Las estadísticas para imputar o escalar se aprenden solo del conjunto de entrenamiento.

## Estado de implementación conservadora

`scripts/auditar_calidad_matriz.py` perfila la matriz maestra consolidada y escribe manifiesto, perfil por columna, cobertura y faltantes por piloto-carrera, distribuciones categóricas, conciliación y registro de excepciones en un directorio nuevo. No escribe otra matriz ni cambia valores. `scripts/normalizar_compound.py` implementa la decisión aprobada en una **nueva matriz histórica** derivada exclusivamente de la maestra: solo las cadenas exactas `"None"` y `"nan"` de `Compound` se convierten a `pd.NA` usando `StringDtype`; cada fila queda registrada con clave completa, literal original, valor resultante, motivo y versión de regla. Genera manifiesto, conciliación y reporte; rechaza sobrescritura.

La normalización no afirma que el vehículo careciera de neumático ni asigna causa. No cambia compuestos físicos, `UNKNOWN`, `TEST_UNKNOWN`, nulos preexistentes, otras columnas, filas ni etiquetas; no incorpora reconstrucciones retrospectivas. La salida no es una matriz de características aprobada para entrenamiento y no sustituye `Compound_asof`, que requiere replay causal y produce `UNKNOWN` si no hay información elegible. Esta conversión no se aplica a actualizaciones parciales de `timing_app_data`: un `None` puede ser campo omitido, por lo que se conserva estado conocido en el mismo stint y no se hereda estado del stint anterior. `Stint_asof` y `FreshTyre_asof` siguen como propuestas separadas.

En el entorno utilizado (Python 3.14 y pandas 2.3.3), los pickles de pandas con `StringDtype` no se pueden volver a leer (`NotImplementedError`). Para conservar una matriz pickle legible, el archivo almacena `Compound` como `object` con valores `pd.NA`; use `load_normalized_matrix(ruta)` de `scripts/normalizar_compound.py` para cargar la matriz con el dtype lógico nullable `string`. La ejecución valida que el resultado recargado mediante ese helper coincide con la matriz normalizada. Para cargar la auditoría CSV conservando el literal `None`, use también `load_normalization_audit(ruta)`; el parser CSV predeterminado de pandas trata esa cadena como marcador NA.

`EDA/dataset/limpiar_matriz.py` conserva la auditoría como modo predeterminado. Se puede invocar la normalización aprobada con `--normalize-compound` y pasar rutas explícitas.

La primera fuente auditada es `vueltas_2022_2025.pkl`, no `vueltas_2022_2025_compound_reconstruido.pkl` ni `vueltas_2022_2025_limpieza_v1.pkl`. Las dos últimas son candidatas derivadas; la revisión encontró que la v1 convierte literales `None` a nulos y no debe usarse como sustituto de la matriz maestra sin una decisión explícita. La auditoría informa saltos, faltantes, valores cero/negativos numéricos y marcas especiales sin borrarlos ni declararlos inválidos por sí solos.

Para ejecutar la auditoría conservadora existente desde la raíz:

```bash
python scripts/auditar_calidad_matriz.py
```

La salida predeterminada es `EDA/dataset/auditorias_limpieza/limpieza_conservadora_v1_3/`. El comando rechaza una carpeta de salida existente. Se puede indicar otra fuente y otra carpeta con `--source` y `--output`. Este resultado es auditoría de calidad, **no** una matriz limpia aprobada ni una matriz de características causales.

La normalización aprobada se reproduce indicando una carpeta de salida nueva:

```bash
python scripts/normalizar_compound.py \
  --input EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025.pkl \
  --output-dir EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_compound_normalizado_v2
```

Para otra ejecución se debe elegir un directorio distinto; no se sobrescriben artefactos. Las decisiones por variable y evidencia vigente se mantienen en [matriz de decisiones de calidad](../analisis/20_matriz_decisiones_calidad_datos.md). El supuesto temporal para FastF1 sigue siendo replay histórico por `Time` con latencia cero supuesta; `Time` no demuestra latencia de recepción en vivo.

Las decisiones por variable y evidencia vigente se mantienen en [matriz de decisiones de calidad](../analisis/20_matriz_decisiones_calidad_datos.md). El supuesto temporal para FastF1 sigue siendo replay histórico por `Time` con latencia cero supuesta; `Time` no demuestra latencia de recepción en vivo.

## Reglas candidatas

| Situación | Tratamiento propuesto | Validación pendiente |
| --- | --- | --- |
| Claves nulas o duplicadas | Marcar como error de integridad; no elegir una fila arbitrariamente. | Definir resolución y pruebas con casos reales. |
| Vueltas faltantes | Registrar discontinuidad por piloto y carrera; no inventar vueltas ni cruzar huecos sin regla. | Investigar comienzos, finales y datos omitidos con fuente independiente. |
| `PitInTime`/`PitOutTime` nulos | Preservar como ausencia de evento registrado; no imputar cero. | Falta de registro no prueba ausencia real del evento. |
| `TyreLife` u otras mediciones faltantes | Conservar faltantes y reportar por carrera y piloto; estudiar imputación y marcador de ausencia en cada experimento. | Confirmar si la información faltante es recuperable y está disponible en el momento de predicción. |
| `Compound` igual exactamente a `"None"` o `"nan"` en matriz maestra consolidada | Convertir a `pd.NA` nullable en una salida histórica derivada nueva; guardar auditoría reversible por fila. | **Acordado e implementado**. No aplica a otros literales, categorías explícitas ni `timing_app_data`. |
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

## Retiro y cierre de la secuencia

El estado histórico final (resultados) puede confirmar un retiro para auditar o cerrar una reproducción, pero solo está disponible al terminar y no debe filtrarse a predicciones anteriores. Para detectar un retiro en vivo, se investigarán de forma conjunta retrasos entre tiempos de sector, mensajes de estado de sesión y continuidad de datos. Los tiempos de sector aislados no bastan: una bandera roja o vuelta neutralizada puede alargar sectores para muchos pilotos; un retraso también puede ser una parada larga o falta de registro.

No hay umbral temporal aprobado. Antes de implementarlo se deben contrastar retiradas, bandera roja, Safety Car/VSC, paradas prolongadas y retrasos de telemetría; definir cuánto esperar y cuántas señales independientes requiere la confirmación. Hasta entonces, las etiquetas de abandono en reproducción histórica deben basarse en el resultado final validado, y las alertas en vivo no deben declarar abandono solo por un sector lento.

## Reconstrucción histórica de datos de neumáticos

La matriz de entrenamiento tendrá una sola fila por piloto y vuelta y un solo valor canónico por variable. No se crearán filas duplicadas ni una matriz paralela con versiones reconstruidas. Se puede investigar una reconstrucción retrospectiva de `Compound` y `TyreLife` usando registros adyacentes del mismo piloto y carrera, restringidos al mismo stint validado y sin cruzar una parada, transición de stint o hueco temporal.

La procedencia (observado o reconstruido), regla y evidencia se conservarán en un informe de auditoría separado, no como variables predictoras por defecto. Si vecinos válidos discrepan, el stint no es seguro o la fuente está ausente en un tramo completo, el valor canónico queda desconocido. La reconstrucción bidireccional solo sirve para auditoría histórica o campos de etiqueta; nunca para la entrada del modelo en un instante anterior al registro que aporta el valor. Para predictores solo se permite reconstrucción causal con datos disponibles hasta el instante de alerta. Validar ocultando datos conocidos y medir recuperación y errores antes de aplicarla.

La validación retrospectiva disponible confirma buen desempeño para valores aislados ocultados, pero también encontró stints enteros sin valores de vuelta que sirvan de referencia. La caché local de `timing_app_data` permitió una reconstrucción controlada de algunos compuestos mediante una correspondencia validada por carrera. El script produce una matriz candidata separada: la matriz maestra permanece intacta, `TyreLife` no se reconstruye y el candidato aún no se considera una versión oficial de entrenamiento. El método, los conteos y sus límites están en [Validación de reconstrucción de datos de neumáticos](../analisis/04_validacion_reconstruccion_neumaticos.md).
