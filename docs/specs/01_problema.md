# Especificación 01: problema y comportamiento de la alerta

**Estado:** borrador para revisión.  
**Alcance:** define el objetivo y el comportamiento esperado; no prescribe una arquitectura única.

## Objetivo

Al cruzar la línea de inicio de una vuelta, estimar la probabilidad de que el piloto realice una parada con cambio de neumáticos en la vuelta que comienza o en alguna de las dos siguientes. Una predicción por encima del umbral configurado activa una alerta.

Ejemplo: al iniciar la vuelta 21, el horizonte comprende las vueltas 21, 22 y 23. El modelo no tiene que predecir cuál de ellas contendrá la parada.

## Información disponible y ventanas

El instante de predicción es el cruce de inicio de la vuelta actual. El modelo recibe resúmenes de telemetría solo de vueltas completadas; no usa telemetría ni tiempo de vuelta de la vuelta que acaba de comenzar. Para clima se alinea la última observación con marca temporal menor o igual al instante de predicción. Si no hay observación anterior disponible, el clima queda desconocido. Esta es una aproximación retrospectiva: la latencia de publicación real del clima aún debe verificarse.

La longitud máxima de ventana **N** es un hiperparámetro y cuenta la observación actual. Al inicio de carrera, la ventana crece con la información disponible hasta alcanzar N; después conserva las N observaciones más recientes. Por ejemplo, con N=8, al inicio de la vuelta 1 hay una observación que contiene únicamente el estado disponible antes de completar vueltas; al inicio de la vuelta 11 se utilizan siete vueltas completas, de la 4 a la 10, más el estado conocido al iniciar la 11. La observación actual no contiene resúmenes calculados con el futuro de esa vuelta. El mecanismo de representación de secuencias cortas se definirá en la especificación de experimentos.

El estado inicial de carrera es la única observación disponible para la primera predicción. Sus variables concretas y disponibilidad son pendientes de investigación.

## Etiquetas históricas y prevención de fuga

El etiquetado se realiza con la información histórica completa: puede consultar lo que ocurrió después del instante de predicción para establecer la respuesta correcta. La etiqueta es booleana cuando existe evidencia suficiente; los casos no determinables se identificarán y tratarán explícitamente.

La etiqueta, `PitInTime`, `PitOutTime` y cualquier variable que revele directamente la respuesta futura se excluyen de las entradas del modelo. Las variables derivadas también se revisarán por disponibilidad temporal y fuga indirecta. La identificación de un cambio de neumáticos exige evidencia adicional a la entrada y salida de boxes; su regla queda pendiente de comprobar con carreras reales.

## Alerta y reproducción

Cada predicción registra su probabilidad, umbral, estado de alerta y horizonte. La aplicación muestra un mensaje de posible cambio de neumáticos durante el horizonte. Si predicciones consecutivas activan la alerta, el mensaje permanece visible y se actualiza el intervalo; si una predicción no la activa, se retira. Se conservan las predicciones individuales para evaluar cada horizonte y las paradas detectadas.

En la reproducción, el usuario selecciona una carrera de prueba y un piloto. El frontend muestra la información real y la alerta del modelo; al observarse una parada real, debe poder comprobar si cayó dentro de algún horizonte previamente alertado. La alerta no atribuye una vuelta exacta ni predice una trayectoria. El frontend será desarrollado por otro integrante; su contrato concreto se acordará posteriormente.

Cuando se confirme un abandono, el piloto deja de recibir predicciones y se comunica el evento. Las señales y el momento de confirmación aún requieren investigación; un retraso entre sectores no basta por sí solo para asumir abandono.

## Criterios de aceptación de la especificación

- El horizonte es la vuelta que comienza y las dos siguientes, sin desfase.
- Las entradas de cada predicción respetan el instante de disponibilidad; ninguna contiene la etiqueta ni datos que revelen el futuro.
- La ventana es adaptativa al inicio y nunca supera N observaciones, incluida la actual.
- Cada alerta conserva su probabilidad e intervalo y puede contrastarse con el evento real.
- Los casos de abandono, inicio de carrera y final de carrera se tratan de acuerdo con reglas documentadas antes de implementarlos.

## Pendientes

Confirmar las variables del estado inicial, las fuentes y latencias disponibles, la detección de cambios de neumáticos y abandonos, el tratamiento del final de carrera y el contrato con el frontend. Definir también cómo evaluar alertas repetidas y seleccionar modelos; F2 y recall por parada son candidatos, no una regla cerrada.
