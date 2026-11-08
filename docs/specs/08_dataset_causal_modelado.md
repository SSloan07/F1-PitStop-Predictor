# SPEC-DATA-001: preparación de datos para experimentos de ML/DL

**Versión:** 0.8.

**Estado:** preparación de datos completada; primera RNN entrenada en Colab y evaluada localmente sobre 2024. Prueba final 2025 reservada.

**Cambio en 0.8:** RNN sencilla como primer experimento, F2 positiva acordada y protocolo Colab/evaluación local en [la especificación 11](11_experimento_rnn_inicial.md). Se actualiza el cierre de preparación train-only.

**Antecedente en 0.7:** se fija la máscara para posiciones históricas no disponibles en ventanas recurrentes, sin desplazar vueltas antiguas. La versión 0.6 especificó el anclaje y elegibilidad temporal en reinicios.

**Alcance:** convertir los datos históricos y las etiquetas candidatas en entradas reproducibles que respeten el instante de predicción, y definir cómo las consumirán los experimentos en Colab/Kaggle.

**Resultado de esta especificación:** define el flujo experimental. Las etiquetas v3, las particiones por carrera y el preprocesador train-only están materializados. La primera RNN F2 fue entrenada y evaluada; ver [resultados](../aclaraciones/06_resultados_rnn_v1_validacion_2024.md).

## 1. Propuesta de trabajo y decisiones vigentes

Construiremos una tabla canónica de características con una fila por piloto y vuelta, y tiempos de disponibilidad para sus valores. El constructor de muestras producirá la entrada de cada decisión al cruce de inicio: solo podrá incorporar información disponible hasta ese instante. La etiqueta se construye por separado con la historia completa. La selección de variables estáticas/dinámicas, el ajuste numérico y la construcción de ventanas pertenecerán a cada experimento configurado.

El trabajo seguirá este orden: fijar contratos y particiones; resolver contradicciones de datos/etiquetas; limpiar con reglas deterministas; reconstruir estados en el tiempo; generar características; publicar una versión candidata; verificarla; ejecutar experimentos en la nube; evaluar con un protocolo común. Una modificación de reglas crea una versión nueva y una comparación documentada.

Se mantienen estas decisiones de la conversación:

- Objetivo operativo acordado en la socialización: **entrada del piloto a boxes durante la carrera bajo condiciones normales, Safety Car o Virtual Safety Car**, con `PitInTime` y `PitOutTime` emparejados del mismo piloto y carrera. Una entrada sin salida o una salida sin entrada queda incierta; una fuente externa no reemplaza la pareja. No se exige demostrar cambio de neumáticos. Los traslados del coche por comisarios no constituyen por sí solos una entrada del piloto. Rige [SPEC-LABEL-002, versión 1.1](07_objetivo_entrada_boxes.md); las referencias antiguas a cambio de neumáticos en las especificaciones 01–03/06 son antecedentes.
- Predicción al cruce que inicia la vuelta `N`; horizonte `N`, `N+1`, `N+2`. Es la misma frontera que terminar la vuelta `N−1` y predecir las tres siguientes. Guardaremos `prediction_lap=N` para evitar ambigüedad.
- Ventana máxima `W` configurable que cuenta la observación actual. Es corta al inicio y crece hasta `W`.
- Cómputo local para preparación, comprobaciones e inferencia/evaluación pequeñas. Entrenamiento y ajuste de modelos en Colab/Kaggle; datos y artefactos versionados en Drive.
- Una sola fila y un solo valor canónico por característica y versión. La procedencia se guarda en auditoría. En el experimento base de la arquitectura combinada, piloto, equipo y circuito forman la rama estática FFNN; las variables que evolucionan durante la carrera forman la rama secuencial RNN. Las comparaciones por ablación pueden quitar una rama, sin cambiar este contrato base.

Estas reglas concretan el [pipeline de datos](02_pipeline_datos.md), el [diseño de experimentos](05_diseno_ortogonal_experimentos.md) y el [perfil de disponibilidad](../analisis/12_perfil_disponibilidad_variables.md). En lo restante, este documento distingue propuestas de decisiones que requieren discusión.

## 2. Punto de partida y asuntos que deben corregirse antes de congelar datos

La matriz histórica inspeccionada es `EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_limpieza_v1.pkl`. El perfil 2022–2023 contiene 47.997 filas y 61 columnas. `etiquetado_pit_entry_v19/labels.pkl` registra 4.990 positivos, 41.361 negativos y 1.646 desconocidos. Estas cifras describen una ejecución anterior a la regla de pareja obligatoria y **no son la distribución esperada de las nuevas etiquetas**. El artefacto v19 no queda aprobado para entrenar bajo el objetivo revisado.

La revisión documental encontró dos asuntos concretos:

1. **Ricciardo, Arabia Saudita 2022, candidato `2022-02-3-IN-36`:** el [informe FIA](https://www.fia.com/news/f1-verstappen-beats-leclerc-thrilling-duel-saudi-arabia) sitúa la detención antes de la entrada al pit lane y describe que los comisarios empujaron posteriormente el auto hacia boxes. Esto contradice el cierre por cruce de fuentes de [la revisión 11](../analisis/11_cierre_revision_eventos_pit_in.md), mientras que [la revisión 10](../analisis/10_revision_pit_in_sin_pit_out.md) ya recogía esa diferencia. La regla nueva exige comprobar si hay una pareja válida y si representa una entrada realizada por el piloto; el traslado de recuperación no basta. Hasta resolverlo, la evidencia no autoriza declarar ese caso definitivamente positivo. Esta especificación no altera v19.
2. **Conteo de fuentes:** el código actual agrupa `official_report_confirms_pit_entry` junto con confirmaciones FIA. Hay cuatro filas con ese estado cuya fuente de revisión es Formula 1. Por ello, el rótulo de 76 confirmaciones «FIA directas» no describe con precisión la procedencia de todas ellas. La siguiente versión deberá distinguir FIA, F1/equipo, fuente secundaria y evidencia del propio feed, sin cambiar una etiqueta solo para corregir el nombre de su fuente.

La política será: prevalece evidencia FIA pertinente al campo discutido; una fuente debe confirmar exactamente lo que se afirma. Las fuentes oficiales permiten auditar eventos y calidad del feed, pero no convierten una entrada sin `PitOutTime` en positivo bajo el contrato 1.1. La ausencia en un resumen de paradas no demuestra que no hubiera una visita para retirarse. Una clasificación final prueba el resultado, pero no la entrada a boxes ni su vuelta. Dos sitios que copian el mismo feed no constituyen dos observaciones independientes. Una discrepancia abierta queda registrada y puede producir `NA`; la meta de la revisión es reducir errores, aunque queden casos sin resolver.

## 3. Contrato temporal de una muestra

La clave única será `(Temporada, Ronda, DriverNumber, prediction_lap)`. Cada muestra guardará `decision_time` en el reloj de la sesión y una referencia al instante UTC cuando exista. Se documentarán unidades y conversión de relojes; no se mezclarán sesiones ni se usarán fechas de modificación de archivos como tiempo de disponibilidad.

Para una característica distinguiremos:

- `event_time`: cuándo ocurrió la medición o terminó el intervalo resumido.
- `available_at`: cuándo pudo recibirse/calcularse su valor.
- `decision_time`: cuándo se produce la predicción.

Se admite un valor si su intervalo de observación terminó y `available_at <= decision_time`. Ningún valor conocido solo después de la decisión puede formar parte de `X`: hacerlo sería fuga de datos. Los desempates con el mismo timestamp tendrán orden de procesamiento declarado: cerrar mediciones recibidas, actualizar estado y emitir la predicción. Una corrección publicada después no puede reescribir una predicción pasada.

Las marcas de evento históricas no prueban la latencia real del feed. Cuando solo exista `event_time`, la ejecución deberá declarar el supuesto de latencia y marcarse como reproducción bajo ese supuesto. Se acordó un modo histórico inicial con **latencia cero supuesta** (`available_at = event_time`), etiquetado como aproximación de replay, y pruebas de sensibilidad con retrasos configurables. La misma configuración de latencia se aplicará en entrenamiento y reproducción de cada experimento. El uso en vivo requerirá comprobar o registrar recepción real; los resultados de latencia cero no se presentarán como rendimiento validado en vivo.

En los 98 límites excepcionales de sesión detectados, el ancla de decisión es el `LapStartTime` que coincide con `Started` tras `Aborted`. La vuelta anterior solo puede aportar características si su `Time` de cierre no supera `decision_time`, bajo el supuesto de latencia configurado. Si `Time` es posterior, esa vuelta queda fuera de la historia de la decisión; pueden usarse vueltas anteriores que sí sean elegibles, manteniendo sus números reales y sin imputar la vuelta faltante. `PitInTime`, `PitOutTime` y agregados retrospectivos de vuelta completa nunca se habilitan como predictores por coincidir con este límite.

Esta propuesta adapta el principio de reconstruir características en un instante pasado: una unión temporal busca hacia atrás, dentro de una antigüedad máxima, y debe considerar también cuándo se publicó una corrección. La documentación de [Feast sobre uniones temporales](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) explica ambas restricciones; adoptar el principio no exige instalar un feature store.

**Ejemplo:** al iniciar la vuelta 21, una observación de clima publicada durante la 21 queda fuera de la decisión. La telemetría completa de la 20 puede usarse solo bajo la regla de disponibilidad/latencia fijada. La entrada a boxes de la 22 puede determinar `y=True`, pero nunca una característica de esa decisión.

Una visita ya iniciada antes de `decision_time` no debe contarse como una entrada futura del mismo horizonte. También se revisarán los cruces de control dentro del pit lane y las salidas sin entrada conocida: una salida confirma presencia en boxes, pero por sí sola no demuestra en qué vuelta comenzó la visita ni cumple la pareja obligatoria. Resolver estos límites forma parte de la auditoría del anclaje temporal de las etiquetas.

## 4. Pipeline con una responsabilidad por etapa

| Etapa | Responsabilidad | Entrada → salida | Ubicación del cómputo |
|---|---|---|---|
| A. Contratos y particiones | Fijar claves, tipos, calendario, reglas y carreras de cada conjunto | Especificación/configuración → esquema y registro de particiones | Local |
| B. Auditoría de etiquetas | Comprobar evento, instante, vuelta, fuente y cobertura del horizonte | Inventario/evidencia → etiquetas nullable versionadas | Local |
| C. Calidad determinista | Detectar marcadores, unidades, incoherencias y canales inválidos sin aprender estadísticas | Datos fuente → valores canónicos y reporte de cambios | Local |
| D. Estado temporal | Reconstruir lo observable antes de cada decisión | Fuentes con timestamps → estados y trazabilidad temporal | Local |
| E. Características | Derivar variables autorizadas con reglas causales | Estados/vueltas completadas → tabla de características | Local |
| F. Publicación | Congelar tablas, esquema, hashes y manifiesto | Características/etiquetas/auditoría → versión de dataset en Drive | Local |
| G. Preparación experimental | Seleccionar columnas, ajustar preprocesadores en train y formar ventanas | Dataset/configuración → entradas del modelo y preprocesador exportable | Local; estado congelado consumido en Colab |
| H. Entrenamiento | Ajustar pesos, guardar checkpoints y registrar cada corrida | Entradas/configuración → modelo y resultados de validación | Colab/Kaggle |
| I. Evaluación y replay | Aplicar el mismo contrato y medir predicciones/alertas | Paquete del modelo/datos → predicciones y métricas | Local si es pequeño; nube si hace falta |

Los notebooks coordinarán estas funciones y mostrarán resultados. La lógica compartida se mantendrá en módulos importables. Para este volumen basta una ejecución secuencial por etapas con reanudación e informes; la propuesta no requiere servicios de orquestación en la nube.

## 5. Registro de características y prevención de fuga

Antes de habilitar una columna se registrarán nombre, significado, tipo, unidad, fuente, clave de entidad, instante/intervalo observado, regla de disponibilidad, antigüedad permitida, tratamiento de faltantes, procedencia de reconstrucción, transformación y grupo experimental. El registro tendrá estados explícitos: `candidate`, `enabled_for_replay`, `validated_for_live` o `excluded`.

La selección de entradas será una **lista de columnas permitidas** por experimento. Una columna nueva no entra automáticamente al modelo. Los identificadores técnicos y la auditoría pueden viajar junto a las muestras, pero no pasan al tensor `X`.

| Familia | Regla propuesta |
|---|---|
| Contexto estático | Piloto, equipo y circuito conocidos antes de la decisión entran en el experimento base de la rama FFNN, con codificación aprendida solo en train y tratamiento de categorías desconocidas. `DriverNumber` es identificador, no magnitud. Temporada/ronda sirven inicialmente para claves y particiones. Se medirá también una ablación sin identidades para detectar cuánto depende el resultado de memorizar estrategias. |
| Estado de neumático | `Compound`, edad, juego y stint solo desde registros cuyo valor esté disponible antes de la decisión. La edad observada debe distinguirse de una edad estimada. No copiar el valor de la primera vuelta del stint a instantes anteriores sin acreditar su disponibilidad. |
| Ritmo y telemetría | Resúmenes de vueltas completadas antes de la decisión. Los datos actuales son agregados por vuelta; un modelo con toda la telemetría por instante requeriría otro contrato y otra representación. No presentar un agregado de acelerador como medida directa de desgaste. |
| Clima | Última observación de esa carrera que cumpla la regla temporal y `K_clima`; conservar timestamp y edad en auditoría. Si expira, el valor queda ausente. Viento se codificará como dirección circular. |
| Safety Car/VSC | Reconstruir el estado en `decision_time` con el historial temporal de estado de pista. Las banderas actuales calculadas sobre todos los códigos de una vuelta contienen lo que sucedió durante ella. Una variable rezagada será nombrada y evaluada como estado de la vuelta anterior. |
| Posición | Utilizar una observación de posición con tiempo de disponibilidad verificable; en una primera variante, la última posición conocida de una vuelta cerrada. No confundirla con la posición definitiva de la carrera. |
| Eventos y auditoría | Excluir `PitInTime`, `PitOutTime`, `PitWithin3Laps`, motivos/fuentes de etiqueta, cobertura futura, estado final del piloto y cualquier derivación del objetivo. Los eventos ya observados pueden cerrar una alerta fuera del modelo, sin exponer eventos futuros. |
| Flags retrospectivas | `Deleted`, `DeletedReason`, `FastF1Generated`, `IsAccurate` y datos similares se usan para diagnóstico/elegibilidad bajo reglas documentadas. No se habilitan como predictores de forma automática. Una vuelta eliminada por límites de pista puede seguir aportando mediciones físicas válidas. |

La prohibición se refiere a información futura respecto a la decisión, no al simple nombre de la columna. Por eso algunos campos de la fila histórica podrán aportar antecedentes tras validarse su tiempo, mientras que sus valores en la vuelta que comienza quedan fuera.

La prevención de fuga y la aplicación consistente del preprocesamiento se apoyan en las prácticas de [scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html): los parámetros de transformación se aprenden en entrenamiento y después se reutilizan al predecir.

## 6. Limpieza y faltantes: dos responsabilidades distintas

**Limpieza local determinista.** Normalizar tipos/unidades, marcadores de ausencia y codificaciones; detectar duplicados, tiempos incoherentes y rangos inválidos. La clave duplicada es un error de integridad, no una fila a elegir al azar. Una regla nueva conserva recuentos, ejemplos y el motivo de cada cambio.

Aplicaremos estas propuestas a los hallazgos actuales antes de aprobar características:

| Hallazgo | Acción propuesta |
|---|---|
| Acelerador 104 y agregados contaminados | Preferir los agregados `_valid_*` o recalcular desde muestras válidas. No dar al modelo las versiones bruta y corregida como si fueran dos mediciones independientes. Conservar cobertura del canal. |
| Marchas máximas mayores que 8 | Revisar el canal original y su diccionario; recalcular todos los agregados afectados desde muestras válidas. Si falta evidencia para recuperar, dejar el canal afectado desconocido, sin recorte silencioso del máximo. |
| Media/máximo de códigos DRS | Excluirlos de la base inicial hasta validar códigos y derivar un estado o proporción con significado. La media aritmética del código no representa apertura física. |
| Velocidad/RPM cero o cantidades extremas de muestras | Revisar intervalos originales, retiros y cobertura. No eliminar toda la fila ni afirmar desgaste por esos valores sin contraste. Documentar si los promedios son por muestras o ponderados por tiempo. |
| Estado de neumático reconstruido | Revisar la auditoría de reconstrucción. Puede ser correcto retrospectivamente y, aun así, no estar disponible a tiempo para el modelo. Los valores no demostrados temporalmente quedan fuera del predictor en ese instante. |

Para predictores se permite recuperar desde observaciones anteriores del mismo coche/carrera, con límites y evidencia de continuidad. No se admite interpolación que use vecinos posteriores al instante de decisión. Tampoco se rellena a través de un cambio de neumáticos o una discontinuidad sin una regla que lo justifique.

**Transformaciones aprendidas en el experimento.** Imputación estadística, escalado, vocabularios/codificadores, selección de variables, reducción dimensional y pesos de clase se ajustan solo con train. La tabla publicada puede estar limpia y contener `NA`; rellenarlos para obtener tensores numéricos forma parte del preprocesador exportado del experimento. Si una característica nunca tiene observaciones válidas en train, se deshabilita o se define un fallback constante explícito, sin consultar validación/prueba.

Se mantendrán tres conceptos distintos: falta de valor de una característica, paso artificial de padding y ausencia de etiqueta. El valor numérico cero no sustituye indiscriminadamente ninguno de ellos. Los indicadores de ausencia/edad son candidatos configurables y deben poder producirse también en replay. Un `NA` de etiqueta se excluye de la función de pérdida y del denominador supervisado, conservando su registro y motivo.

## 7. Particiones antes de aprender estadísticas

La división materializada es **2022–2023 para entrenamiento, 2024 para validación y 2025 para prueba final**, con etiquetas generadas bajo el mismo contrato. El manifiesto y las listas de `sample_id` por conjunto están en `muestras_causales_v1_3_etiquetas_v3_particiones_v1_1`; las carreras completas se mantienen juntas. La cobertura por temporada y carrera acompaña la publicación.

Todas las vueltas y todos los pilotos de una carrera permanecerán en un solo conjunto. Dos ventanas solapadas de un piloto, o pilotos de la misma carrera expuestos al mismo Safety Car, no deben quedar repartidos por una partición aleatoria de filas. La validación interna opcional en 2022–2023 será por bloques cronológicos de carreras completas, entrenando con bloques anteriores y validando con posteriores. En cada bloque se reajusta el preprocesador desde su train.

La correlación temporal y por grupos requiere particiones que la respeten; la documentación de [validación cruzada de scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html) describe ambas condiciones. Aquí se aplicarán juntas a carreras completas; ordenar filas y usar un splitter temporal sin agrupar carreras no cumple el contrato.

Las relaciones entre características y objetivo, selección de columnas y estimación de umbrales estadísticos de calidad se investigarán con train. Las reglas físicas/del feed quedan versionadas y se aplican de forma idéntica a todos los conjuntos. La revisión técnica previa ya inspeccionó calidad en 2025: se dejará constancia, sin describir prueba como un conjunto jamás observado. A partir de ahora no se usarán sus métricas de modelo para diseñar experimentos ni ajustar alertas.

## 8. Ventanas adaptativas sin desfase

Una muestra en la vuelta `N` tendrá hasta `W−1` posiciones históricas y un token de estado actual. La secuencia lógica contiene `min(W, N)` posiciones; se almacena en un tensor de forma fija `W` usando padding cuando falten posiciones iniciales. `W` cuenta el presente; el horizonte de etiqueta permanece fijo en 3 aunque se cambie `W`. La predicción se emite desde la primera vuelta: no hay un mínimo de historial ni se espera a completar `W`. Por ejemplo, al iniciar la vuelta 2 con `W=10`, el modelo predice usando el estado disponible de la vuelta 1 y el estado actual de la 2; las otras ocho posiciones son padding enmascarado.

La ventana tiene forma fija de `W` posiciones y conserva la identidad de cada vuelta. Si una vuelta histórica dentro del rango configurado no había terminado al llegar `decision_time`, esa posición se marca con la máscara de disponibilidad temporal; no se desplazan vueltas más antiguas para llenarla ni se imputan sus valores. El token actual permanece en su posición y contiene solo estado conocido al inicio de la vuelta. En las primeras vueltas, las posiciones que aún no existen se rellenan como padding con una máscara de padding separada. Las máscaras de ausencia temporal, padding y faltante de una característica se distinguen entre sí. Por ejemplo, para la decisión de la vuelta 9 con `W=8`, las posiciones son vueltas 2–8 más el estado actual de la 9; si la vuelta 8 aún no era elegible, su posición queda enmascarada y la vuelta 1 no se corre a ese lugar.

| Decisión | `W` | Información dinámica permitida |
|---|---:|---|
| Inicio de vuelta 1 | 8 | Solo estado conocido antes/al inicio. No hay telemetría de vuelta completada. |
| Inicio de vuelta 2 | 8 | Resumen causal de vuelta 1 y estado actual de vuelta 2. |
| Inicio de vuelta 3 | 8 | Resúmenes de vueltas 1–2 y estado actual de vuelta 3. |
| Inicio de vuelta 11 | 8 | Siete vueltas completas, 4–10, y estado actual de vuelta 11. |

### Ejemplo temporal para revisar juntos

Los horarios de este ejemplo son **ilustrativos**, no corresponden a una carrera del dataset. Supongamos que el piloto inicia la vuelta 11 a las `14:35:00`, que la ventana configurada es `W=8` y que la eventual entrada a boxes ocurre en la vuelta 12, con salida emparejada. El constructor forma siete posiciones históricas, vueltas 4–10, más una posición de estado actual para la 11. La etiqueta de la decisión vale `True` por la pareja `PitInTime`/`PitOutTime` de la vuelta 12, pero esos registros no aparecen en ninguna rama de entrada.

| Dato del ejemplo | Instante en que queda disponible | Entrada al iniciar la vuelta 11 | Motivo |
|---|---|---|---|
| Piloto, equipo y circuito de esta carrera | Antes de la salida | FFNN estática | Contexto conocido antes de decidir. |
| Ritmo/telemetría agregada de vueltas 4–9 | Como máximo `14:35:00` | RNN, tokens históricos 4–9 | Vueltas terminadas y datos ya recibidos. |
| Resumen de telemetría de vuelta 10 | `14:35:03` | Ausente en el token histórico 10 de esta decisión | Aunque la vuelta terminó, el resumen llegó después de decidir. No se sustituye por el resumen de la 3 ni se corre la ventana. |
| Observación meteorológica | `14:34:48` | Estado actual de RNN, si cumple `K_clima` | Última observación anterior disponible; se registra su edad. |
| Observación meteorológica siguiente | `14:35:12` | Excluida | Se recibió después de decidir. |
| Mensaje de Safety Car/VSC recibido | `14:34:58` | Estado actual de RNN | El estado era conocido en ese instante. |
| Cambio de estado de pista recibido | `14:35:05` | Excluido | Pertenece al futuro de esta decisión. |
| Sectores, tiempos y telemetría completa de vuelta 11 | Después de `14:35:00` | Excluidos | La vuelta apenas comienza. |
| `PitInTime`/`PitOutTime` de la vuelta 12 | Después de `14:35:00` | Solo etiqueta y auditoría | Confirman el resultado real; nunca forman parte de `X`. |

Si un estado de neumáticos proviene de una actualización publicada después de `14:35:00`, también queda ausente aunque una reconstrucción posterior permita conocerlo. En la vuelta 1 no hay tokens históricos: la RNN recibe únicamente el estado inicial efectivamente disponible. En la vuelta 2 recibe el token de la 1 y el estado actual; así crece hasta `W` sin alterar el horizonte de tres vueltas.

El caso de la vuelta 10 distingue **posición de ventana** de **valor disponible**: el token temporal existe, pero una característica tardía dentro de él queda ausente con su máscara y motivo. La política de latencia por fuente determinará ese `available_at`; el ejemplo no presume que los archivos históricos registren la hora real de recepción.

Cada token histórico `j` identifica su vuelta y contiene mediciones de esa vuelta solo si ya terminó y sus datos estaban disponibles al decidir en `N`. Los campos nombrados como «estado al inicio de j» conservan ese significado; no se rellenan con información posterior a ese inicio. El token actual `N` tiene ausentes sus agregados de vuelta completa. Las derivaciones se calculan sobre las observaciones permitidas, nunca con una media, máximo o normalización de toda la carrera futura.

Por eso la tabla canónica distinguirá **estado al inicio de j** y **mediciones de j completada**, con sus tiempos de disponibilidad. Puede conservar ambas familias en una misma fila histórica sin duplicar observaciones. Esa fila no se entrega completa a `X` cuando se decide al inicio de j: el constructor oculta sus mediciones de vuelta completa. Después, al decidir en una vuelta posterior, esas mediciones podrán formar parte de un token histórico si ya estaban disponibles. Esta distinción mantiene la ventana de siete vueltas completas más presente del ejemplo, sin añadir por accidente otra vuelta de historial mediante un rezago adicional.

El constructor validará el reloj, además del número de vuelta. Mantendrá el orden temporal y no cruzará piloto, sesión, carrera o partición. Un hueco interno no se resolverá tomando las siguientes filas ni extendiendo la ventana a vueltas más antiguas sin declararlo. Para la primera versión proponemos reportar/rechazar las ventanas con discontinuidad estructural; los faltantes parciales de características se representan por su política de imputación y máscara. La adaptación corta está aprobada para el comienzo de carrera, no es una reparación implícita de huecos intermedios.

Para formar batches se propone padding al final, longitud real y máscara explícita de pasos. La máscara de valores ausentes se trata aparte. La documentación de [TensorFlow sobre padding y masking](https://www.tensorflow.org/guide/keras/understanding_masking_and_padding) explica cómo omitir pasos artificiales; todas las capas elegidas deberán respetar la máscara. No se inferirá padding porque una fila escalada resulte llena de ceros.

La ausencia de etiqueta de una muestra no obliga a borrar sus mediciones históricas válidas de otras ventanas; elegibilidad de `y` y validez del contexto se evalúan por separado. Los retiros solo cerrarán predicciones a partir de información disponible en ese momento. El resultado final puede auditar la reproducción, pero no anticipar cuándo dejar de predecir.

## 9. Artefactos de datos y contrato de consumo

La publicación propuesta contendrá los siguientes archivos o equivalentes con esquema explícito. Los nombres son contratos futuros, no comandos/artefactos ya implementados.

| Artefacto | Contenido |
|---|---|
| `features.parquet` | Una fila por piloto-vuelta con clave, inicio, cierre y características canónicas candidatas. Distingue estado inicial y mediciones completadas con disponibilidad por campo/familia. Es fuente del constructor de muestras, no una entrada irrestricta a `fit`. Sin normalización aprendida globalmente. |
| `labels.parquet` | Clave, objetivo booleano nullable, motivo y versión de etiqueta. Se une uno a uno, verificando cobertura y ausencia de multiplicación de filas. |
| `feature_registry.yaml` | Lista, orden, tipos, unidades, disponibilidad, grupos y reglas de cada característica; exclusiones explícitas. |
| `splits.csv` | Una carrera por fila con temporada, identificador, fecha y partición fijada. |
| `quality_report.json` | Cobertura antes/después, faltantes por carrera/piloto/canal, anomalías, exclusiones y ejemplos revisables. |
| `lineage.parquet` | Trazabilidad de valores reconstruidos, fuentes, tiempos y reglas. Uso de auditoría; no se incorpora automáticamente a `X`. |
| `manifest.json` | Versiones/hashes de entradas y salidas, código/especificación/configuración, fuentes, supuestos de latencia y estado de aceptación. |

Parquet es la propuesta de intercambio entre local, Colab y Kaggle; se comprobará que preserve tipos, `NA`, unidades y precisión temporal. La fuente pickle se conserva para trazabilidad. Las ventanas se construirán con el mismo módulo/configuración cuando el experimento las solicite; no se necesita almacenar una copia de la matriz por cada `W`. Un caché opcional deberá identificar hashes de datos, código y configuración.

La validación de la unión con etiquetas detiene la publicación si hay claves duplicadas, filas sin correspondencia inesperadas o variaciones de cardinalidad. La salida informa todas las exclusiones, positivas y negativas, por motivo; no oculta cambios de prevalencia mediante descartes.

## 10. Organización propuesta según SRP

La organización final se adaptará al repositorio sin mover todo su contenido como requisito de esta fase. La separación de responsabilidades propuesta es:

| Ubicación futura | Responsabilidad |
|---|---|
| `src/f1_pitstop/contracts/` | Esquemas, claves y validaciones de contratos. |
| `src/f1_pitstop/quality/` | Reglas deterministas de calidad y sus informes. |
| `src/f1_pitstop/alignment/` | Relojes, disponibilidad y uniones temporales. |
| `src/f1_pitstop/features/` | Derivaciones de características, sin entrenar modelos. |
| `src/f1_pitstop/datasets/` | Unión controlada, particiones, ventanas y máscaras. |
| `src/f1_pitstop/preprocessing/` | Transformadores ajustables en train y serialización. |
| `src/f1_pitstop/models/` | Definiciones de baseline y arquitecturas. |
| `src/f1_pitstop/evaluation/` | Métricas y política de evaluación de alertas compartida. |
| `src/f1_pitstop/inference/` | Carga del paquete, actualización de estado e inferencia. |
| `configs/data/`, `configs/experiments/` | Decisiones versionadas; sin parámetros ocultos en celdas. |
| `scripts/` | Entradas de ejecución delgadas que llaman los módulos. |
| `notebooks/templates/`, `notebooks/experiments/` | Plantilla remota y un `.ipynb` identificable por experimento. |
| `docs/specs/`, `docs/analisis/` | Contratos propuestos/aprobados y resultados de inspecciones. |

## 11. Experimentos reproducibles en Colab/Kaggle

Cada experimento tendrá pregunta, hipótesis, configuración base y factor que cambia, siguiendo la [especificación de ortogonalización](05_diseno_ortogonal_experimentos.md). La arquitectura combinada de referencia tendrá dos entradas: una FFNN para piloto, equipo y circuito, y una RNN para la secuencia temporal de neumáticos, ritmo, telemetría agregada disponible, clima y estados de carrera. Sus representaciones se fusionarán antes de la salida binaria. El modelo recibirá solo la porción de secuencia disponible al instante de decisión, con ventana adaptable y máscaras. Un baseline tabular sencillo, una RNN sin rama estática y las ablaciones de grupos/ventana servirán de comparación; no se presupone que la arquitectura combinada sea superior.

Los modelos comparables usarán mismas particiones, población evaluable y contrato de tiempo. El baseline tabular tendrá acceso causal documentado a los grupos elegidos; si difiere la información o el resumen del historial, se registrará como un factor adicional. Las comparaciones de clima, identidad, telemetría y arquitectura se harán mediante ablaciones declaradas.

Cada `.ipynb` podrá ejecutarse desde un runtime limpio, de arriba abajo, con rutas de entrada/salida parametrizadas. Cargará una versión concreta del paquete compartido y del dataset, verificará sus hashes y ajustará preprocesadores solo con train. Las dependencias se fijarán al implementarse; se guardarán versión de Python, framework, CUDA/GPU y semillas. Una ejecución se identifica por experimento, configuración y semilla. Varias semillas comparten la misma lista entre modelos comparables; se propone al menos tres en la comparación final si el presupuesto lo permite.

Las semillas y algoritmos deterministas reducen variación, pero no garantizan igualdad numérica entre hardware o versiones. Así lo documenta [PyTorch sobre reproducibilidad](https://docs.pytorch.org/docs/stable/notes/randomness.html). Fijaremos el entorno y reportaremos variabilidad; el determinismo entre Colab y Kaggle no se prometerá solo por usar la misma semilla.

Los pesos de clase se estiman desde train. Si se estudia remuestreo, se aplicará únicamente a ejemplos de entrenamiento ya construidos, conservando su orden interno; validación/prueba mantienen la prevalencia natural. La configuración registrará pérdida, optimizador, scheduler, regularización, batch size, ventana, selección de checkpoint y presupuesto máximo. Early stopping observa validación y exporta el checkpoint seleccionado por la regla previa, no una época elegida mirando prueba.

La salida exportada incluirá pesos, definición/hiperparámetros de arquitectura, lista y orden de características, preprocesador ajustado, categorías desconocidas, máscaras, configuración temporal, umbral, calibrador si se utiliza, historial, métricas, predicciones y manifiesto. Guardar solo pesos no permite reconstruir el experimento. Para reanudar entrenamiento tras una desconexión se guardarán además optimizador, época y estados aleatorios necesarios.

## Estado de datos y siguiente paso

La ejecución de etiquetas vigente es `etiquetado_pit_entry_v3_2022_2025`, generada con la regla de pareja obligatoria y el mismo horizonte N–N+2. El corte 2022–2023 es idéntico a `etiquetado_pit_entry_v2_pair_required` en las 47.997 claves comparadas. Las ventanas causales conservadas son `muestras_causales_v1_3`; su instantánea antigua de etiquetas no se consume.

La partición publicada en `muestras_causales_v1_3_etiquetas_v3_particiones_v1_1` mantiene carreras completas: train 2022–2023, validation 2024 y test final reservado 2025. Su configuración es `configs/particion_pit_entry_v1.json`; las tablas contienen pertenencia y elegibilidad supervisada, y el manifiesto fija hashes de los datos referenciados. La prueba final no se consulta para decisiones de diseño experimental.

El contrato de preprocesamiento train-only está implementado en [SPEC-DATA-003](10_contrato_preprocesamiento.md). El artefacto vigente se publica en `muestras_preprocesadas_v1_3`; medias, escalas y vocabularios se ajustan solo con `train AND supervised_eligible`, y las transformaciones se aplican a train/validación con el estado congelado. No se generan transformaciones de prueba. La primera RNN seleccionó checkpoint y umbral en validación 2024, con pesos exportados y paridad local comprobada; 2025 permanece reservada.

El primer experimento acordado es una **RNN sencilla**, implementada con una capa recurrente de 32 unidades y una rama estática mínima. Su configuración, notebook de Colab, exportación de pesos y evaluación local están definidos en [SPEC-EXP-001](11_experimento_rnn_inicial.md). Esa referencia ya se ejecutó y evaluó localmente. La siguiente comparación mantiene arquitectura y entrenamiento y cambia el criterio de selección a F1, según [la especificación 12](12_experimento_rnn_f1.md). La MLP y otras arquitecturas quedan para comparaciones posteriores, sin ser un requisito previo.

El evaluador y el replay cargarán ese mismo paquete y las mismas funciones. Las nuevas categorías se transformarán con una política de desconocidos fijada desde train. Una predicción guardará clave, instante, score/probabilidad, estado de calibración, umbral, horizonte, versión de modelo y alerta. La primera implementación debe poder procesar un piloto y mantener estados aislados si después se procesan varios simultáneamente; el desarrollo del frontend queda a cargo del compañero.

## 12. Evaluación propuesta y elección de umbral

Se acordó **maximizar F2 de la clase positiva** como criterio inicial para seleccionar modelo/umbral en validación, conservando precision, recall, F1, Average Precision y matriz de confusión. F2 favorece recall respecto a precision; su definición está en [scikit-learn, F-beta](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html). Este criterio no fija un presupuesto aceptable de falsas alertas: su costo permanece visible en las métricas auxiliares.

El protocolo inicial selecciona checkpoint por F2 a umbral fijo 0,5, con menor BCE de validación como desempate. Después congela los pesos y busca umbral en 0,05–0,95 con paso 0,01; desempata por mayor precision y luego mayor umbral. La cobertura por evento se reporta por separado. La búsqueda y cualquier calibración usarán validación, sin ajustar de nuevo sobre prueba. Un protocolo posterior podría maximizar recall sujeto a una precision mínima o límite de alertas falsas, pero esos valores deben acordarse y versionarse antes de medir prueba. Reportar solo recall permitiría una alerta permanente, por lo que siempre se mostrará también su costo en falsas alertas.

Se medirán dos unidades distintas:

- **Piloto-vuelta:** precision, recall, F1/F2, Average Precision y falsos positivos por cada 100 decisiones con etiqueta evaluable. Los `NA` no se convierten en negativos para calcular métricas. Informar su cantidad y los horizontes censurados.
- **Evento real:** proporción de entradas evaluables cubiertas por al menos una alerta emitida antes del evento y cuyo horizonte lo incluya. Cada evento se cuenta una sola vez, aunque haya tres alertas previas. Informar antelación de la primera alerta válida, eventos sin oportunidad de predicción y cobertura total.

Se conservará cada predicción individual aunque el frontend agrupe alertas consecutivas en un mensaje. La cobertura por evento no se llamará precision por episodio. La política de persistencia/expiración de mensajes se fijará antes de calcular métricas de episodios.

Se publicarán métricas agregadas y por carrera/temporada, incluyendo condiciones de SC/VSC, lluvia y entradas para retiro cuando haya evidencia suficiente. Las ventanas solapadas no son observaciones independientes para estimar incertidumbre: si se calculan intervalos, el remuestreo será por carreras completas, además del resumen entre semillas. Un evento sin oportunidades evaluables se informa con su motivo; no desaparece del reporte para mejorar recall.

## 13. Criterios de aceptación antes de entrenar

Estas verificaciones se implementarán en la etapa de construcción; no se ejecutan como parte de este documento.

| Verificación | Resultado exigido |
|---|---|
| Claves y unión | Claves únicas, unión uno a uno y todas las diferencias de cobertura explicadas. |
| Futuro alterado | Borrar o modificar datos posteriores a `decision_time` no cambia `X` de esa decisión; `y` sí puede cambiar porque usa futuro. |
| Batch frente a replay | Construcción de características/ventanas sobre prefijos produce los mismos valores y máscaras que el procesamiento cronológico de la carrera. |
| Vuelta actual | En el inicio de N no aparecen telemetría completa, sectores finales ni estado agregado de pista de N. |
| Clima | Una observación posterior no se selecciona; al superar `K_clima` queda `NA` con edad/motivo conservados. |
| Correcciones históricas | Una reconstrucción apoyada en una observación posterior no entra como predictor previo. |
| Caso temporal de boxes | Evento en N etiqueta N−2, N−1 y N cuando son decisiones anteriores al ingreso; no etiqueta N+1 por ese evento. Entrada/salida emparejadas no duplican visitas. |
| Etiqueta desconocida | Final incompleto sin positivo verificable conserva `NA`; no se usa en pérdida/métricas como negativo. |
| Ventana corta | N=1 contiene solo estado inicial; N=11/W=8 contiene completas 4–10 más presente 11; padding no se confunde con un cero válido. |
| Aislamiento | Ninguna ventana, imputación o estado se mezcla entre pilotos/carreras/particiones; reportar huecos. |
| Ajuste estadístico | Cambiar validación/prueba no cambia parámetros aprendidos del preprocesador de train. |
| Evidencia de eventos | Resolver contradicciones como Ricciardo 2022 y corregir la categoría de fuentes antes de liberar etiquetas. |
| Consumo remoto | Runtime limpio ejecuta el notebook con entradas versionadas y exporta un paquete recargable por el evaluador. |

Los fallos de integridad o fuga temporal bloquean la publicación para entrenamiento. Las limitaciones conocidas, latencias supuestas y características no habilitadas se enumeran en el manifiesto. Pasar estas comprobaciones acredita el contrato medido; no prueba que el modelo vaya a tener buen recall ni que el feed futuro tenga idéntica calidad.

## 14. Estado y siguiente etapa

1. **Completado:** contratos de objetivo y tiempo causal.
2. **Completado:** etiquetas v3 de 2022–2025, con invariancia verificada para 2022–2023.
3. **Completado:** muestras causales y ventanas/máscaras v1_3 verificadas y conservadas.
4. **Completado:** alineación de etiquetas v3 y particiones cronológicas por carrera, con prueba final reservada.
5. **Completado:** preprocesamiento train-only publicado en `muestras_preprocesadas_v1_3`, con configuración y estado congelados.
6. **Implementado:** primer experimento RNN sencillo, F2 positiva, notebook de Colab y exportación/evaluación local; ver especificación 11.
7. **Completado:** referencia F2 entrenada en Colab y evaluación local sobre validación. **Preparado:** ensayo F1 de la especificación 12. Prueba final 2025 reservada.

## 15. Puntos para socializar y cerrar

| Punto | Propuesta para discutir | Por qué debe fijarse |
|---|---|---|
| Alcance exacto de visita | Acordado: carrera normal, Safety Car o Virtual Safety Car; solo `PitInTime` y `PitOutTime` válidos y emparejados. Un registro incompleto queda incierto. Auditar traslados asistidos, cruces dentro del pit lane y entradas ya iniciadas antes de decidir. | Fija objetivo, anclaje temporal y qué alertas son útiles. |
| Latencia | Acordado: ningún dato posterior a la decisión entra en `X`; primer replay con latencia cero **supuesta** y pruebas con retrasos configurables. Registrar recepción real antes de afirmar capacidad en vivo. | Un timestamp de evento no acredita recepción inmediata. |
| `K_clima` | Acordado: base de 120 segundos tras la [revisión de cobertura 2022–2023](../analisis/17_revision_estado_pista_clima.md); sigue configurable. Registrar segundos y mostrar minutos. | Controla antigüedad y ausencia de clima sin elegir el valor con validación/prueba. |
| Bandera roja | Acordado: pausar alertas mientras el estado conocido sea rojo, reportar esas decisiones y auditar reinicios/cruces por boxes. | Una suspensión no representa una vuelta ordinaria de predicción. |
| Primera vuelta | Mantener muestra de estado inicial con campos realmente disponibles y faltantes explícitos. | Debe poder inferirse sin usar el cierre de la primera vuelta. |
| Ventana y máscaras | `W` cuenta presente; usar W=8 como ejemplo de especificación, y elegir el valor base de entrenamiento en configuración. Padding y faltantes de valores con máscaras diferentes. | Evita desfases y permite el experimento adaptativo aprobado. |
| Identidad estática | Acordado: piloto/equipo/circuito en la FFNN del experimento base; variables que cambian con la carrera en la RNN. Comparar por ablación con y sin contexto estático. | Aísla valor del contexto y riesgo de memorización. |
| Selección de modelos | Acordado: F2 positiva por muestra; checkpoint y umbral según especificación 11, con recall por evento y falsas alertas visibles. | Hace comparable la búsqueda y evita elegir criterio mirando prueba. |
| Reserva final | Mantener 2025 para evaluación final y declarar la inspección técnica previa. Cualquier cambio del protocolo debe registrarse. | Permite interpretar honestamente qué generalización mide el resultado. |

La socialización deberá producir una versión aprobada de estas decisiones y un primer encargo de implementación acotado. Los hallazgos nuevos se registrarán con el dato, su impacto y la versión que corrige el problema; no se ajustarán reglas silenciosamente entre notebooks.
