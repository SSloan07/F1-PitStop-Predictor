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

En esa agrupación, la fuente de timing app muestra más de un valor no nulo de `Compound` en **295 stints** y más de un valor no nulo de `New` en **187** a lo largo del tiempo. Esto describe actualizaciones dentro de un stint, no necesariamente conflictos simultáneos. La reconstrucción retrospectiva ya aplicada a algunos compuestos se conserva como auditoría; no se retropropaga automáticamente a `X` de vueltas anteriores.

## Reproducción as-of al inicio de cada vuelta

Se reprodujo la secuencia de eventos por `(Driver, Stint, Time)` para las **47.997 decisiones** del corte. En cada decisión se identifica primero el último índice de stint ya reportado; luego se toma el último valor disponible de los campos dentro de ese stint activo, con `Time <= LapStartTime`. El índice fuente se convierte a numeración de matriz sumando uno. Este orden evita arrastrar `Compound` o `New` del stint anterior. Es un replay por tiempo de evento con supuesto de latencia cero, no una garantía de recepción en vivo. La antigüedad de la última actualización tiene mediana **88,1 s**, p90 **117,2 s**, p99 **841,7 s** y máximo **7.576,2 s**; un estado de neumático puede persistir, así que estos datos no justifican por sí solos un TTL fijo.

La instantánea as-of no siempre coincide con el valor retrospectivo que FastF1 deja en la fila actual:

| Estado temporal de la caché | Comparación contra la fila consolidada actual | Casos distintos |
|---|---:|---:|
| `Compound_asof` | Coincide en 46.543 de 47.997 decisiones (97,0%) | 1.454; 1.254 tienen `PitOutTime` en la fila actual y 200 no. |
| `New_asof` frente a `FreshTyre` | Coincide en 47.029 de 47.997 (98,0%) | 968; 476 tienen `PitOutTime` actual y 492 no. |
| `Stint_asof` (índice fuente + 1) frente a `Stint` matriz | Coincide en 45.422 de 47.997 (94,6%) | 2.575; 1.672 tienen `PitOutTime` actual y 903 no. |

Una diferencia en la fila actual puede ser información posterior a la decisión, así que no prueba por sí misma que el as-of esté mal. Persisten **200 diferencias de `Compound`** y **492 de `New`/`FreshTyre`** incluso cuando la fila actual no tiene `PitOutTime`; deben reportarse y no resolverse con el valor final retrospectivo. Hay **1.813 grupos** con el mismo piloto y `Time`; **1.741** contienen varios índices de stint porque FastF1 aplana un snapshot con varias entradas de stint en varias filas con el mismo tiempo. No son conflictos del campo `Stint`. Al reagrupar por `(Driver, Stint, Time)`, no se encontraron valores simultáneos contradictorios de `Compound` ni de `New`. El orden original del pickle no se asumirá como tiempo real de recepción.

La documentación oficial de FastF1 define `Stint` como contador de stint y `New` como indicador de si el neumático era nuevo al montarse; el parser representa cada entrada de stint del snapshot en una fila con el mismo timestamp ([parser de TimingAppData](https://github.com/theOehrly/Fast-F1/blob/main/fastf1/_api.py#L3476-L3626), [definiciones de campos](https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/timing_data.rst#L187-L217)). Por eso, varias filas al mismo tiempo se interpretan como una instantánea desglosada, no como evidencia automática de conflicto.

`New` cambia a lo largo del tiempo en **187 de 2.588 stints**: 162 transiciones `False→True` y 25 `True→False`. En 144 de las primeras transiciones, el primer `Compound` era `UNKNOWN`; en 150, `TotalLaps` era cero. En las 25 transiciones inversas, `TotalLaps` inicial también era cero. Estos patrones sugieren registros iniciales incompletos, pero no prueban su causa: los valores iniciales no interpretables permanecerán `UNKNOWN`. El último `New` as-of coincide con el `FreshTyre` final de matriz en **2.560/2.588 stints (98,9%)**; esa concordancia final no autoriza a completar decisiones pasadas con el valor final.

Para `TyreLife`, la fuente de timing app no trae un campo homónimo; contiene `TotalLaps` y `StartLaps`, cuya equivalencia no se asumirá. La última fila previa consolidada aporta `TyreLife` en **46.568** decisiones donde existe una vuelta previa elegible bajo `Time <= decision_time` y latencia cero; en las demás no se sustituirá por el valor de la fila actual ni se inferirá la edad desde `FreshTyre`. Las 18 decisiones con cierre previo posterior siguen la regla de la [máscara temporal de la especificación](../specs/08_dataset_causal_modelado.md#8-ventanas-adaptativas-sin-desfase).

## Decisión propuesta por columna

| Columna | Propuesta para la RNN | Condición para habilitarla |
|---|---|---|
| `Stint_asof` | **Propuesta: candidata dinámica.** Contador del stint ya reportado al instante de decisión; no equivale al número de paradas. | Replay causal del índice de stint activo conocido hasta la decisión. Revisar los seis inicios sin pareja contigua; incluir solo si el estado puede conocerse sin mirar el futuro. Ablacionar junto con `TyreLife_last_eligible` por posible redundancia. |
| `Compound_asof` | **Candidata aprobada para el baseline:** última categoría publicada al llegar la decisión, con `UNKNOWN` explícito. | Usar la unión temporal de `timing_app_data` bajo la latencia declarada; invalidar si hay empate categórico sin resolución o cambio de stint sin compuesto verificable. No usar la fila actual ni reconstrucciones futuras. |
| `TyreLife` (`TyreLife_last_eligible`) | **Candidata aprobada para el baseline:** edad numérica de la última vuelta cerrada y disponible, con máscara propia. | Solo usar la fila previa si `Time <= decision_time` y el valor está observado; conservar ausencias de stint. No tomar `TyreLife` de la vuelta actual ni inferir desde `TotalLaps`, `StartLaps` o `FreshTyre`. |
| `FreshTyre_asof` | **Propuesta: reservar para ablación posterior, no para el baseline inicial.** Booleano derivado del último `New` interpretable del juego activo; puede aportar cuando la edad falta, pero se solapa con `TyreLife`. | Replay as-of por stint. Mantener `UNKNOWN` ante registros iniciales incompletos; nunca copiar el `FreshTyre` final hacia atrás ni inferirlo de `TyreLife`. |

El constructor de ventanas mantendrá separadas las mediciones de neumáticos de vueltas completadas y el **estado actual conocido**. Si una actualización de compuesto/juego llega durante la vuelta `N`, solo podrá afectar decisiones posteriores a su `available_at`; no reescribe `X` de la decisión al inicio de `N`. Si los registros disponibles no permiten demostrar cuándo se conoció un valor, quedará desconocido en esa decisión con una máscara distinta del padding temporal. El estado actual se reconstruirá solo desde el último evento elegible; los campos de la fila consolidada que reflejan una actualización posterior no rellenarán posiciones históricas.

## Comprobaciones para la futura transformación

1. Por piloto-carrera, stint ordenado y valores coherentes; reportar excepciones por carrera, nunca corregirlas silenciosamente.
2. Cada característica de neumático tendrá fuente, `event_time`, `available_at` o supuesto de latencia declarado, y momento de decisión.
3. Reproducir una carrera prefijo a prefijo dará el mismo estado que construirla en batch sin permitir que el futuro cambie un prefijo.
4. Una corrección publicada después de una decisión no alterará el compuesto, edad ni frescura vistos por ese prefijo.
5. Los 35 stints sin `TyreLife` observado conservarán ausencia salvo evidencia causal independiente; las 36 filas sin `Compound` conservarán procedencia y política temporal explícitas.
6. Cambios de neumático o de stint nunca generan por sí solos `y=True`; solo la pareja válida `PitInTime`/`PitOutTime` lo hace.

**Decisión socializada (2026-11-07):** se aprobaron `Compound_asof` y `TyreLife_last_eligible` como candidatas del baseline, con las reglas anteriores. Esto no aprueba su materialización ni elimina la necesidad de reproducirlas as-of. Para socializar ahora: incluir `Stint_asof` como candidata, sujeto a replay temporal y ablación; reservar `FreshTyre_asof` para una ablación posterior, con `UNKNOWN` para estados iniciales incompletos. Ninguna de estas dos propuestas está aprobada todavía. Después se pasa a tiempos de vuelta, sectores, velocidades de timing y posición.
