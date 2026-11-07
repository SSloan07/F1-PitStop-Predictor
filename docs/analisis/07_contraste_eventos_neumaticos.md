# Contraste inicial de eventos de neumáticos

**Estado:** muestra dirigida preliminar; no valida una regla general ni autoriza etiquetas definitivas. Solo se utilizaron casos de 2022; 2025 permanece reservado para prueba.

Se contrastó la matriz `vueltas_2022_2025_limpieza_v1.pkl` con informes de carrera y equipo. La selección buscó un cambio al mismo compuesto, cambios de compuesto y un caso con convención de vuelta dudosa; no es una muestra representativa. Los informes de F1 pueden compartir información de timing con FastF1, por lo que no se presupone independencia completa.

| Caso | Evidencia histórica en matriz | Contraste | Resultado |
| --- | --- | --- | --- |
| Bahréin 2022, Verstappen, entrada vuelta 14 | `PitInTime` en 14, `PitOutTime` en 15, stint 1→2; SOFT→SOFT | La [tabla oficial de paradas](https://www.formula1.com/en/results/2022/races/1124/bahrain/pit-stop-summary) registra parada en 14; la [crónica oficial](https://www.formula1.com/en/latest/article/leclerc-leads-1-2-for-ferrari-in-bahrain-gp-as-both-red-bulls-suffer-late.3GyOQOvMIMA8FIH17cXOoT) describe otro juego blando, aunque llama vuelta 15 a la parada. | Evidencia de cambio al mismo compuesto; la asignación de vuelta requiere declarar la convención de entrada. |
| Mónaco 2022, Albon, entrada vuelta 18 | Entrada en 18, salida en 19; stint 1→2 y WET→HARD | [Williams](https://www.williamsf1.com/articles/082bc817-39e8-45b1-9f7f-4130f446ea0a/2022-monaco-grand-prix) indica cambio a duro en vuelta 18. | Concordancia de entrada y cambio. |
| Mónaco 2022, Albon, entrada vuelta 22 | Entrada en 22, salida en 23; stint 2→3 y HARD→MEDIUM | El mismo informe de Williams describe un pinchazo y cambio a medio cuatro vueltas después del cambio en 18. | Concordancia con un cambio posterior; la crónica no da explícitamente el número 22 para la segunda parada. |
| Mónaco 2022, Latifi, primeras vueltas | Entrada en 1 y salida en 2; WET→INTERMEDIATE. También hay entrada en 2 y salida en 3. | Williams sitúa el cambio a intermedios en vuelta 2. | Existe un desfase o ambigüedad de atribución: no se debe asignar automáticamente sin reconciliar tiempos y convenciones. |

Estos casos respaldan seguir investigando la combinación de entrada, salida y transición de stint, pero no prueban su precisión general. La discrepancia entre tabla y crónica para Verstappen y el caso Latifi muestran que una mención de vuelta en una crónica no equivale necesariamente a `LapNumber` de entrada. No se estimó cobertura ni exactitud poblacional.

Faltan contrastes de entradas sin cambio de neumáticos, cambios retrasados de stint, bandera roja, abandonos y horizontes incompletos, además de criterios de aceptación y fuente independiente para cada caso. Hasta resolverlos no se generan etiquetas booleanas definitivas ni se interpretan eventos ambiguos como negativos.
