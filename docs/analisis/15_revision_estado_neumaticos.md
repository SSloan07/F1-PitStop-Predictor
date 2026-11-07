# Revisión 2: estado de neumáticos disponible al predecir

**Fuente principal:** `EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_limpieza_v1.pkl`, corte 2022–2023 de 47.997 filas. Se inspeccionaron también las 44 cachés locales `timing_app_data.ff1pkl` de esas carreras; no se realizaron llamadas a la API. Esta revisión desarrolla `Stint`, `Compound`, `TyreLife` y `FreshTyre` del [inventario](13_inventario_columnas_para_modelado.md). El [contrato temporal](../specs/08_dataset_causal_modelado.md) sigue siendo obligatorio: solo información disponible al inicio de la vuelta puede entrar en la RNN.

**Estado:** análisis para socializar. No altera datos ni etiquetas, no crea características y no aprueba automáticamente estas cuatro columnas como entradas.

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

## Decisión propuesta por columna

| Columna | Propuesta para la RNN | Condición para habilitarla |
|---|---|---|
| `Stint` | Estado dinámico o número de cambios observados hasta la decisión, tratado como recuento, no como evidencia de una futura parada. | Construir desde registros temporalmente válidos. Si el cambio solo se detecta retrospectivamente, el estado actual queda desconocido; revisar los seis inicios sin pareja contigua. |
| `Compound` | Último compuesto conocido antes de la decisión, como categoría con `UNKNOWN`. | Fuente/valor con tiempo verificable, manejo de correcciones y expiración si se define. No usar el `Compound` de la fila actual ni la reconstrucción futura por defecto. |
| `TyreLife` | Edad observada del juego actual, con indicador de ausencia; una edad derivada sería una característica distinta. | Partir de una observación causal y comprobar continuidad del mismo juego. No reiniciar a 1 por cambio de stint ni extrapolar a través de un cambio no verificado. |
| `FreshTyre` | Estado nuevo/usado del juego actual, booleano con desconocido explícito. | Respetar el instante de disponibilidad; no inferirlo de `TyreLife` ni asumir que cero faltantes en la matriz significa cobertura en vivo. |

El constructor de ventanas mantendrá separadas las mediciones de neumáticos de vueltas completadas y el **estado actual conocido**. Si una actualización de compuesto/juego llega durante la vuelta `N`, solo podrá afectar decisiones posteriores a su recepción; no reescribe `X` de la decisión al inicio de `N`. Si los registros disponibles no permiten demostrar cuándo se conoció un valor, quedará fuera de esa decisión o se marcará como supuesto de replay, con una máscara distinta del padding temporal.

## Comprobaciones para la futura transformación

1. Por piloto-carrera, stint ordenado y valores coherentes; reportar excepciones por carrera, nunca corregirlas silenciosamente.
2. Cada característica de neumático tendrá fuente, `event_time`, `available_at` o supuesto de latencia declarado, y momento de decisión.
3. Reproducir una carrera prefijo a prefijo dará el mismo estado que construirla en batch sin permitir que el futuro cambie un prefijo.
4. Una corrección publicada después de una decisión no alterará el compuesto, edad ni frescura vistos por ese prefijo.
5. Los 35 stints sin `TyreLife` observado conservarán ausencia salvo evidencia causal independiente; las 36 filas sin `Compound` conservarán procedencia y política temporal explícitas.
6. Cambios de neumático o de stint nunca generan por sí solos `y=True`; solo la pareja válida `PitInTime`/`PitOutTime` lo hace.

**Siguiente grupo propuesto:** tiempos de vuelta, sectores, velocidades de timing y posición. Antes de implementarlo falta resolver el contrato de disponibilidad de neumáticos y los seis cambios de stint excepcionales.
