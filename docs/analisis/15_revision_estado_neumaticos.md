# Revisión 2: estado de neumáticos disponible al predecir

**Fuente principal:** `EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_limpieza_v1.pkl`, corte 2022–2023 de 47.997 filas. Se inspeccionaron también las 44 cachés locales `timing_app_data.ff1pkl` de esas carreras; no se realizaron llamadas a la API. Esta revisión desarrolla `Stint`, `Compound`, `TyreLife` y `FreshTyre` del [inventario](13_inventario_columnas_para_modelado.md). El [contrato temporal](../specs/08_dataset_causal_modelado.md) sigue siendo obligatorio: solo información disponible al inicio de la vuelta puede entrar en la RNN.

**Estado:** análisis para socializar. No altera datos ni etiquetas ni crea características. `Compound_asof` y `TyreLife_last_eligible` están aprobadas como candidatas del baseline, sujetas a las reglas temporales de esta revisión; `Stint_asof` y `FreshTyre_asof` siguen pendientes.

## Perfil observado en la matriz

| Columna | Faltantes | Resultado de consistencia | Límite para el modelo |
|---|---:|---|---|
| `Stint` | 0 | Valores enteros de 1 a 8. Las 874 secuencias piloto-carrera comienzan en 1; hay 1.714 aumentos, todos de una unidad y ninguno negativo. | El número de stint de una fila histórica no acredita cuándo se publicó el cambio. |
| `Compound` | 36 | Cinco categorías: HARD, MEDIUM, SOFT, INTERMEDIATE y WET. No cambió entre dos valores conocidos de un mismo stint. | La estabilidad retrospectiva no autoriza usar el compuesto de la vuelta que empieza si llegó después. |
| `TyreLife` | 555 | En 44.889 pares consecutivos con edad conocida dentro del mismo stint, aumentó exactamente una unidad. | No equivale necesariamente a vueltas corridas en carrera; un juego usado puede empezar con edad mayor que 1. |
| `FreshTyre` | 0 | Verdadero en 36.972 filas y falso en 11.025; permanece constante dentro de cada uno de los 2.588 stints. | No hay marca temporal propia en la fila consolidada; no es prueba de que ya fuera conocido al iniciar cada vuelta. |

Hay **2.588 stints**. `Compound` falta en 35 filas de un stint completo de Mónaco 2022 y en una fila aislada de Austria 2022. `TyreLife` falta en **35 stints completos**, concentrados en Austria 2022 (359 filas), Mónaco 2022 (161) y Canadá 2023 (35). Un valor vecino dentro del mismo stint no basta para reconstruir estos bloques porque no existe ancla observada de `TyreLife` allí. La [revisión anterior de reconstrucción](04_validacion_reconstruccion_neumaticos.md) explica por qué una recuperación retrospectiva de huecos aislados no valida rellenar stints enteros ni usar vecinos futuros en una predicción histórica.

`FreshTyre` y `TyreLife` tampoco son equivalentes: tomando el primer registro disponible de cada stint, hay 46 stints con `FreshTyre=False` y `TyreLife=1`, y un stint con `FreshTyre=True` y `TyreLife>1`. No rellenaremos una columna a partir de la otra. Se conservan ambos valores para auditar su origen y significado antes de decidir reglas de calidad.

## Cambios de stint y eventos de boxes

De los **1.714** inicios de stint posteriores al primero, **1.708** tienen `PitOutTime` en la fila nueva y `PitInTime` en la fila anterior. Los otros **seis** inicios no tienen `PitOutTime` en la fila nueva ni `PitInTime` en la anterior: uno en Mónaco 2022, cuatro en Austria 2022 y uno en Canadá 2023. Son cambios de estado históricos que no se explican con una pareja contigua de entrada/salida en esta matriz; no se convertirán por ello en etiquetas positivas ni en un cambio de neumático conocido en tiempo real. Requieren inspección de timing antes de habilitar el estado actual de esos stints.

El [objetivo aprobado](../specs/07_objetivo_entrada_boxes.md) exige `PitInTime` y `PitOutTime` emparejados para contar una visita. `Stint`, el compuesto o la edad de neumático **no sustituyen** esos registros al construir `y`. A la inversa, la existencia de una pareja de boxes no vuelve automáticamente disponibles las características del nuevo neumático en el instante de decisión.

## Qué enseña la caché temporal

Las 44 cachés contienen al menos un registro de timing app por cada uno de los 2.588 stints de la matriz al alinear el índice de stint de esa fuente con el de la matriz. Para **923 stints**, el primer `Time` registrado en la caché es anterior o igual al `LapStartTime` de la primera vuelta del stint; para **1.665**, es posterior. Es una comparación entre marcas de tiempo de la fuente, **no una medición de latencia real de recepción**. Que el registro tenga hora temprana tampoco prueba que su valor definitivo estuviera publicado entonces.

En esa misma agrupación, la fuente de timing app muestra más de un valor no nulo de `Compound` en **295 stints** y más de un valor no nulo de `New` en **187**. La matriz consolidada, en cambio, registra `Compound` y `FreshTyre` constantes dentro de cada stint. Por ello, no se seleccionará una etiqueta de la caché por stint sin revisar el orden de eventos, correcciones y concordancia con los valores observados. La reconstrucción retrospectiva ya aplicada a algunos compuestos se conserva como auditoría; no se retropropaga automáticamente a `X` de vueltas anteriores.

## Reproducción as-of al inicio de cada vuelta

Se hizo una unión temporal hacia atrás por piloto para las **47.997 decisiones** del corte: para cada `LapStartTime`, se tomó el último registro de `timing_app_data` con `Time <= LapStartTime`, conservando el orden original de registros en empates y arrastrando el último valor no nulo de cada campo. La unión encuentra valores en las 47.997 decisiones, pero esta cobertura no demuestra recepción en vivo: `Time` solo permite un replay por tiempo de evento bajo latencia cero supuesta. La antigüedad de la última instantánea tiene mediana **88,0 s**, p90 **116,7 s**, p99 **778,3 s** y máximo **7.576,2 s**; un estado de neumático puede persistir, así que estos datos no justifican por sí solos un TTL fijo.

La instantánea as-of no siempre coincide con el valor retrospectivo que FastF1 deja en la fila actual:

| Estado temporal de la caché | Comparación contra la fila consolidada actual | Casos distintos |
|---|---:|---:|
| `Compound` | Coincide en 46.510 de 47.997 decisiones (96,9%) | 1.487; 1.253 tienen `PitOutTime` en la fila actual y 234 no. |
| `New` frente a `FreshTyre` | Coincide en 47.004 de 47.997 (97,9%) | 993; 476 tienen `PitOutTime` actual y 517 no. |
| `Stint` fuente + 1 frente a `Stint` matriz | Coincide en 45.711 de 47.997 (95,2%) | 2.286; la numeración de fuente parece iniciar en 0 y la matriz en 1, pero el desplazamiento no resuelve todos los casos. |

Una diferencia en la fila actual puede ser precisamente información posterior a la decisión, así que no se tratará como prueba de error del as-of. Sin embargo, las **234 diferencias de `Compound`** y **517 de `New`/`FreshTyre`** aun cuando no hay `PitOutTime` en esa fila impiden dar por reconciliados los campos. Hay **1.813 grupos** con el mismo piloto y `Time` en la caché; **1.741** tienen más de un valor de `Stint`, cuatro más de un `Compound` y seis más de un `New`. El orden original del pickle no se asumirá como tiempo de recepción real. Los conflictos categóricos se marcarán desconocidos hasta fijar una regla determinista de desempate respaldada por la fuente.

Para `TyreLife`, la fuente de timing app no trae un campo homónimo; contiene `TotalLaps` y `StartLaps`, cuya equivalencia no se asumirá. La última fila previa consolidada aporta `TyreLife` en **46.568** decisiones donde existe una vuelta previa elegible bajo `Time <= decision_time` y latencia cero; en las demás no se sustituirá por el valor de la fila actual ni se inferirá la edad desde `FreshTyre`. Las 18 decisiones con cierre previo posterior siguen la regla de la [máscara temporal de la especificación](../specs/08_dataset_causal_modelado.md#8-ventanas-adaptativas-sin-desfase).

## Decisión propuesta por columna

| Columna | Propuesta para la RNN | Condición para habilitarla |
|---|---|---|
| `Stint` | Estado dinámico o número de cambios observados hasta la decisión, tratado como recuento, no como evidencia de una futura parada. | Construir desde registros temporalmente válidos. Si el cambio solo se detecta retrospectivamente, el estado actual queda desconocido; revisar los seis inicios sin pareja contigua. |
| `Compound_asof` | **Candidata aprobada para el baseline:** última categoría publicada al llegar la decisión, con `UNKNOWN` explícito. | Usar la unión temporal de `timing_app_data` bajo la latencia declarada; invalidar si hay empate categórico sin resolución o cambio de stint sin compuesto verificable. No usar la fila actual ni reconstrucciones futuras. |
| `TyreLife` (`TyreLife_last_eligible`) | **Candidata aprobada para el baseline:** edad numérica de la última vuelta cerrada y disponible, con máscara propia. | Solo usar la fila previa si `Time <= decision_time` y el valor está observado; conservar ausencias de stint. No tomar `TyreLife` de la vuelta actual ni inferir desde `TotalLaps`, `StartLaps` o `FreshTyre`. |
| `FreshTyre` | Estado nuevo/usado del juego actual, booleano con desconocido explícito. | Respetar el instante de disponibilidad; no inferirlo de `TyreLife` ni asumir que cero faltantes en la matriz significa cobertura en vivo. |

El constructor de ventanas mantendrá separadas las mediciones de neumáticos de vueltas completadas y el **estado actual conocido**. Si una actualización de compuesto/juego llega durante la vuelta `N`, solo podrá afectar decisiones posteriores a su `available_at`; no reescribe `X` de la decisión al inicio de `N`. Si los registros disponibles no permiten demostrar cuándo se conoció un valor, quedará desconocido en esa decisión con una máscara distinta del padding temporal. El estado actual se reconstruirá solo desde el último evento elegible; los campos de la fila consolidada que reflejan una actualización posterior no rellenarán posiciones históricas.

## Comprobaciones para la futura transformación

1. Por piloto-carrera, stint ordenado y valores coherentes; reportar excepciones por carrera, nunca corregirlas silenciosamente.
2. Cada característica de neumático tendrá fuente, `event_time`, `available_at` o supuesto de latencia declarado, y momento de decisión.
3. Reproducir una carrera prefijo a prefijo dará el mismo estado que construirla en batch sin permitir que el futuro cambie un prefijo.
4. Una corrección publicada después de una decisión no alterará el compuesto, edad ni frescura vistos por ese prefijo.
5. Los 35 stints sin `TyreLife` observado conservarán ausencia salvo evidencia causal independiente; las 36 filas sin `Compound` conservarán procedencia y política temporal explícitas.
6. Cambios de neumático o de stint nunca generan por sí solos `y=True`; solo la pareja válida `PitInTime`/`PitOutTime` lo hace.

**Decisión socializada (2026-11-07):** se aprobaron `Compound_asof` y `TyreLife_last_eligible` como candidatas del baseline, con las reglas anteriores. Esto no aprueba su materialización ni elimina la necesidad de reproducirlas as-of. `Stint_asof` y `FreshTyre_asof` siguen pendientes por las discrepancias y los seis inicios de stint sin pareja contigua. Después se pasa a tiempos de vuelta, sectores, velocidades de timing y posición.
