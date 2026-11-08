# Estado actual y plan de limpieza histórica

**Estado:** fase de limpieza histórica cerrada hasta el límite de las decisiones acordadas; matriz preservada y registro de calidad reproducible publicado.
**Etapa:** preparación y limpieza histórica de datos. La construcción de entradas temporales y los experimentos corresponden a etapas posteriores.

## Base revisada y comprobaciones realizadas

Se inspeccionó `EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_compound_normalizado_v1/vueltas_2022_2025_compound_normalizado.pkl`. Su SHA-256 coincide con el manifiesto: `02c15d7b2d766e1d01604cf143bb288a2e584370f0f422766683866a10ca00e0`. La fuente maestra registrada es `9f8eb3f40d66869778c30d75f0a1ebc45196b8ad03c3a4258cc8f5e335dab65c`.

La matriz tiene 101.290 filas, 61 columnas, 92 carreras y 1.826 secuencias piloto-carrera. La auditoría conservadora se repitió sobre esta salida en un directorio temporal: validación estructural/esquema aprobada, sin errores. No se encontraron claves nulas, duplicadas ni saltos internos de vuelta según sus controles. Esto no demuestra cobertura completa de toda la carrera o publicación oportuna de cada dato.

Pasaron las 17 pruebas actuales de normalización y auditoría. El normalizador ya exige evidencia de hash y recuentos antes de escribir. La salida v1 se conserva con su manifiesto histórico; no se regeneró por haber cambiado posteriormente el código de validación.

El repositorio continúa en `bc8fd06`, con nueve archivos rastreados modificados y nuevos scripts/documentos/artefactos locales. Este diagnóstico no modifica esos cambios. La base funcional de limpieza existe y no debe reconstruirse desde cero.

## Diagnóstico por familia

| Familia | Evidencia actual | Implicación para la limpieza |
|---|---|---|
| Identidad y claves | Sin faltantes en Driver, Team o GranPremio; claves válidas. | Conservar y verificar como condición obligatoria de cada ejecución. |
| Compound | 780 ausencias reales después de las 780 conversiones aprobadas. | Regla cerrada. No reabrir su semántica ni incorporar silenciosamente reconstrucciones históricas. |
| TyreLife | 989 ausencias; edades presentes enteras de 1–78. | La política ya acordada es conservar ausencias, sin imputar desde otros campos. |
| Stint | 354 ausencias, todas en los bloques iniciales de Miami 2025 ya investigados. Índices presentes enteros de 1–8, sin decrementos en las secuencias revisadas. | Conservar y registrar; no completar como conteo de paradas. |
| FreshTyre | Sin nulos, pero True en las 354 filas de Miami sin estado de neumáticos previo en caché. | No confundir cobertura booleana con evidencia semántica. Mantener la limitación en auditoría. |
| Vueltas generadas y posición | 135 filas generadas; exactamente las mismas 135 filas tienen Position ausente. | Ausencia explicada por esta condición; no imputar posición ni borrar las secuencias. |
| LapTime y sectores | 384 LapTime ausentes, 2.094 Sector1Time ausentes. Ninguna duración presente revisada es cero o negativa. | Conservar los faltantes y separar los casos antes de cualquier reconstrucción. |
| SpeedFL | 3.602 ausencias: 3.467 con PitInTime y 135 en filas generadas. | Faltantes estructurales/contextuales; no rellenar cero ni eliminar las vueltas de boxes. |
| Telemetría Speed/RPM | 38 filas con velocidad media/máxima cero y 39 con RPM media/máxima cero. | Revisión dirigida de trazas por canal; cero por sí solo no autoriza invalidación. |
| Acelerador filtrado | 1.080 filas con alguna muestra fuera de 0–100; 40 sin muestras válidas. Conteos, fracciones y agregados filtrados son coherentes en toda la matriz. | Mantener los agregados filtrados existentes y los 40 faltantes. Formalizar controles; no recalcular todo sin una discrepancia concreta. |
| Marcha y DRS | 41 máximos de marcha mayores que 8; DRS contiene códigos de estado. | Conservar en histórico; su exclusión/pausa del baseline ya está documentada. No recortar agregados ni promediar códigos como magnitudes interpretables. |
| Clima | Sin nulos en la matriz, pero Weather_Time es posterior al inicio de vuelta en 101.263 filas. | Es un producto retrospectivo por vuelta. Conservar; la selección as-of se construye después desde caché. No modificar timestamps para aparentar disponibilidad. |

### TyreLife puede cerrarse sin rellenarlo

Los 989 faltantes se distribuyen así: Mónaco 2022, 161; Austria 2022, 359; Canadá 2023, 35; Australia 2025, 5; Miami 2025, 354; Bélgica 2025, 75.

Hay 635 filas en 41 stints identificados completamente sin edad conocida (633 no generadas y dos generadas), más 354 filas en 15 bloques piloto-carrera sin stint identificado. No hay stints identificados con una mezcla de edades presentes y ausentes que permita una reparación por vecinos. Entre vueltas consecutivas con edad conocida del mismo stint, los incrementos observados son +1.

La acción necesaria es materializar su registro de calidad y las comprobaciones de conservación. Mantener los 989 faltantes satisface la política existente; su presencia no bloquea toda la limpieza ni exige una nueva aprobación de imputación.

### Casos que sí merecen investigación dirigida

- Speed/RPM: BOT, Japón 2022, vueltas 3–28; MAG, Miami 2022, vuelta 1; LEC, Azerbaiyán 2022, vuelta 21 (RPM); HAD, Australia 2025, vuelta 1 generada; ALB, Italia 2025, vueltas 1–10. Los patrones no son idénticos: el bloque de ALB tiene acelerador cero, no 104. No extender una regla de un caso a todos los canales o carreras.
- Sectores: 99.093 filas tienen LapTime y tres sectores; 14 presentan una diferencia de 0,504–0,518 s entre LapTime y la suma, todas en 2024 ronda 22 y con IsAccurate=False. En 2022–2023 la igualdad es exacta en las 46.867 filas completas. Revisar fuente y contexto antes de imponer igualdad global o corregir componentes.
- Reconstrucciones algebraicas: 1.773 filas sin Sector1Time tienen los otros tres componentes necesarios y una diferencia positiva; 858 pertenecen a 2022–2023. Son candidatas, no mediciones recuperadas automáticamente. La positividad no valida por sí sola la reconstrucción.
- Reloj entre vueltas: 22 cierres de vuelta anterior son posteriores al inicio registrado de la siguiente (18 en Australia 2023 y cuatro en Mónaco 2024). Ninguna fila individual tiene Time anterior a su propio LapStartTime. La revisión v2 vincula cada inicio con `Started` posterior a `Aborted`; no corregir timestamps y no usar el cierre tardío de la vuelta previa en el historial causal.

La inspección de 2024–2025 describe calidad y coherencia de origen. No se entrenaron modelos ni se usaron sus métricas para escoger reglas estadísticas o configuraciones.

## Ejecución del plan y resultado actual

Se ejecutó el plan sobre la matriz histórica normalizada v1 ya publicada, sin regenerarla ni duplicarla. El proceso `scripts/generar_registro_calidad_historica.py` valida hashes de la matriz, manifiesto, auditoría de Compound y cachés consultadas; comprueba contrato de claves, esquema, faltantes de neumáticos y acelerador, y publica solo informes en `EDA/dataset/auditorias_limpieza/registro_calidad_historica_v1/`.

Artefactos: `quality_register.csv` (registro por clave completa y campo), `variable_quality_summary.csv` (61 variables), `source_case_review.csv` (cinco casos de telemetría), `sector_source_review.csv` (casos de Las Vegas 2024), `clock_boundary_review.csv`, `reconciliation.json`, `report.md` y `manifest.json`. La salida `registro_calidad_historica_v1` se conserva como historial, pero su resumen de reloj tenía una asociación de carrera incorrecta; para conclusiones de solapamientos usar v2. Reproducción desde la raíz, con una carpeta de salida nueva:

```bash
python -m scripts.generar_registro_calidad_historica \
  --input EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_compound_normalizado_v1/vueltas_2022_2025_compound_normalizado.pkl \
  --manifest EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_compound_normalizado_v1/manifest.json \
  --compound-audit EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_compound_normalizado_v1/compound_normalization_audit.csv \
  --cache-root EDA/dataset/cache_fastf1_carreras \
  --output-dir EDA/dataset/auditorias_limpieza/registro_calidad_historica_v3
```

La carpeta de salida debe ser nueva; el script rehúsa sobrescribir una existente. El nombre `v3` del ejemplo es solo ilustrativo: usar otra ruta nueva si ya existe. La salida corregida v2 validó 101.290 filas, 61 columnas, cero claves nulas/duplicadas, 16.644 anotaciones de calidad, 780 ausencias de Compound reconciliadas, 989 TyreLife y 354 Stint faltantes preservados, 14 discrepancias de sectores coincidentes exactamente con la caché extendida y 22 solapamientos temporales evaluados individualmente. La coincidencia con la caché confirma consistencia con la fuente FastF1 local, no validación independiente FIA.

Los ceros de telemetría investigados coinciden con trazas locales, pero su significado físico o de feed sigue sin determinarse; por eso se conservan. En las 14 filas con discrepancia entre vuelta y suma de sectores, `IsAccurate=False` y la caché extendida reproduce los mismos cuatro valores. La matriz identifica la ronda 3 de 2023 como Australia; la caché correspondiente y `session_info` identifican Australian Grand Prix, no Arabia Saudita. Las 18 transiciones australianas y las cuatro de Mónaco 2024 tienen `LapStartTime` exactamente coincidente con `Started` inmediatamente posterior a `Aborted` (tolerancia de 1 ms); los mensajes Race Control cercanos incluyen bandera roja y reanudación. El v1 consultaba por error la caché de Arabia Saudita y queda supersedido para esta conclusión. Se conservan timestamps y el cierre de la vuelta previa, posterior al momento de decisión, no puede alimentar el historial causal. `SpeedFL` tiene dos filas PitIn con valor presente que se marcan para revisión; las ausencias observadas están contenidas en el contexto PitIn o fila generada.

La clasificación de acelerador queda alineada con las decisiones existentes: `Telemetry_Throttle_valid_mean/max` son candidatas RNN acordadas para vueltas cerradas elegibles; `Telemetry_Throttle_valid_samples/fraction` y `Telemetry_Samples` son solo cobertura/auditoría y quedan fuera de las entradas del baseline inicial. Esto no aprueba variables nuevas.

La publicación no escribió ni alteró matriz, etiquetas o cachés; tampoco imputó ni corrigió valores. `Stint_asof` y `FreshTyre_asof` siguen como propuestas separadas. `Compound_asof`, las características temporales causales, las decisiones finales sobre casos ambiguos y cualquier entrenamiento/evaluación quedan fuera de esta fase. Las validaciones focalizadas del registro, la auditoría y la normalización se ejecutan con `python -m unittest scripts.test_generar_registro_calidad_historica scripts.test_auditar_calidad_matriz scripts.test_normalizar_compound`.

## Plan de acción propuesto

### 1. Fijar la base y el registro de calidad

Usar la salida normalizada v1 y sus hashes como punto de partida de esta revisión. Mantener la maestra y las cachés como evidencia original. Conservar el procedimiento actual de no sobrescritura y exigir evidencia apropiada a cada entrada/etapa.

Crear un registro separado, enlazado por clave completa y campo, con motivos como `missing_entire_identified_stint`, `missing_stint_in_cached_initial_block`, `generated_row`, `no_valid_throttle_samples` o `sector_sum_discrepancy`. Un caso puede tener varios motivos. Incluir evidencia, acción, versión y estado de revisión; no agregar automáticamente estos motivos como predictores.

**Aceptación:** todas las ausencias/anomalías contempladas por el contrato tienen explicación o estado explícito; la matriz conserva las filas y valores originales salvo las transformaciones aprobadas.

### 2. Cerrar y automatizar las políticas ya decididas

- TyreLife/Stint: conservar los 989 y 354 faltantes respectivamente; verificar que no se introdujo imputación. Registrar los bloques y la limitación de FreshTyre de Miami.
- Acelerador: mantener 0 <= muestras válidas <= muestras totales, fracción coherente con su cociente, media/máximo filtrados en 0–100 y ausencia exactamente cuando no hay muestras válidas. Conservar los agregados brutos como auditoría.
- SpeedFL, Position y campos Pit: conservar las ausencias contextuales. Un PitInTime/PitOutTime nulo no significa medición numérica cero.
- Banderas FastF1Generated/IsAccurate: conservar y asociar a su contexto, sin filtrado global de filas.

**Aceptación:** controles y casos focalizados reproducibles; ningún cambio de valores físicos por conveniencia; no exigir cero faltantes como criterio de calidad.

### 3. Resolver la lista corta de anomalías de fuente

Comenzar por las trazas Speed/RPM de 2022–2023 ya investigadas. Verificar intervalo de selección y comparar con ejemplos normales, de boxes y de suspensión. Ampliar la auditoría de coherencia a los otros casos de origen sin derivar umbrales del rendimiento de modelos.

Revisar las 14 discrepancias de sectores y los 22 límites temporales con sus eventos de sesión. Reconciliar las diferencias con los informes antiguos que cubrían solo 2022–2023. No tratar automáticamente una interrupción como reloj incorrecto ni la bandera IsAccurate=False como invalidez de todos los campos.

Para cada caso emitir una decisión por campo: conservar, marcar para revisión o proponer invalidación/reparación con evidencia. Si hay una propuesta nueva que altere valores, presentarla con claves y recuentos concretos antes de materializarla. La reconstrucción de sectores queda como trabajo opcional posterior mientras no haya una regla validada. En v2, la identidad de carrera y las 22 transiciones temporales quedaron conciliadas contra cachés locales con `session_info`, estado de sesión y mensajes Race Control; el solapamiento se conserva como límite de disponibilidad, no como error de timestamp.

**Aceptación:** cada caso investigado tiene evidencia local y resolución o motivo para conservarlo. Cualquier invalidación afecta únicamente al campo/canal justificado, conserva el original en auditoría y mantiene las filas.

### 4. Publicar una versión histórica con contrato verificable

Componer las reglas aprobadas en un flujo ejecutable: entrada versionada → validación → transformaciones → conciliación → publicación. Reutilizar las funciones existentes y mantener la evidencia de cada etapa. No volver a aplicar la CLI de normalización con recuentos de la maestra a una entrada intermedia cuyos marcadores ya desaparecieron.

Publicar una nueva matriz solo cuando haya cambios de valores aprobados. Si el resultado de una revisión consiste en conservar valores y añadir evidencia, publicar el reporte/registro de calidad sin duplicar innecesariamente la matriz.

**Aceptación:** mismas claves/orden/filas; cambios por campo conciliados; fuentes intactas; reglas reversibles cuando corresponda; hashes y manifiesto; pruebas de rutas válidas y de rechazos; lectura con los helpers documentados; comando de reproducción. Conservar los artefactos previos.

## Trabajos que ya tienen salida o pertenecen a la siguiente etapa

La candidata `etiquetado_pit_entry_v2_pair_required` ya existe con 4.817 positivas, 41.361 negativas y 1.819 desconocidas. Se comprobó que sus 47.997 claves emparejan uno a uno con el corte 2022–2023 de la matriz normalizada y que su manifiesto identifica la maestra original. La normalización exclusiva de Compound no obliga a regenerar esas etiquetas. Las 64 entradas inciertas mantienen la regla de pareja requerida; no es necesario forzar su clasificación para terminar la limpieza histórica.

`Compound_asof`, `TyreLife_last_eligible`, clima/estado de pista as-of, máscaras y ventanas se implementan en la construcción posterior de muestras. `Stint_asof` y `FreshTyre_asof` conservan su estado de propuestas separadas; las decisiones de selección de variables no bloquean conservar la tabla histórica.

Marcha/DRS no necesitan una inversión inmediata para el primer baseline porque ya están en pausa/excluidos. Los imputadores, escaladores y selección estadística de umbrales se ajustan después en entrenamiento, dentro de los experimentos.

## Resultado esperado al cerrar esta fase

Una versión histórica identificable con reglas aprobadas, evidencia por campo, ausencias conservadas y anomalías explicadas o señaladas. Puede conservar valores faltantes y casos inciertos y aun así cumplir su contrato de limpieza. El siguiente hito será construir las entradas causales, no continuar indefinidamente intentando completar todas las celdas.

Referencias: [matriz de decisiones](../analisis/20_matriz_decisiones_calidad_datos.md), [revisión de neumáticos](../analisis/15_revision_estado_neumaticos.md), [ritmo y sectores](../analisis/16_revision_ritmo_sectores_posicion.md), [clima/pista](../analisis/17_revision_estado_pista_clima.md), [telemetría](../analisis/18_revision_telemetria_agregada.md), [normalización de Compound](03_evaluacion_marcadores_compound.md).
