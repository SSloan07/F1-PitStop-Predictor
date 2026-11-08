# Revisión de eventos `PitInTime` sin `PitOutTime`

**Artefacto revisado:** `etiquetado_pit_entry_v2`  
**Temporadas:** 2022–2023  
**Alcance:** revisar los eventos sin salida emparejada y los candidatos que no figuran en el resumen de paradas FIA. No se cambian etiquetas en este informe.

## Resultado del inventario

El artefacto contiene 63 eventos sin resolver: 59 `PitInTime` sin `PitOutTime` y cuatro candidatos ausentes del resumen de paradas FIA. Los 63 ocurren en la última vuelta disponible para ese piloto. Esto describe el límite de los datos; por sí solo no demuestra que el piloto entrara al pit lane ni que abandonara fuera de él.

Se conserva `unknown` para los eventos terminales sin evidencia individual que confirme o descarte la visita. Que un piloto se retire, deje de producir telemetría o no figure en el resumen de paradas no basta para inferir la ubicación del auto.

## Candidatos no listados en el resumen FIA

| Caso | Evidencia revisada | Decisión para el dataset |
|---|---|---|
| Verstappen, Baréin 2022, v.54 | El resumen FIA lista sus paradas de servicio en v.14, v.30 y v.43. El informe oficial de F1 relata que volvió a los boxes y se retiró en la vuelta 54. | La visita está respaldada por la crónica oficial de F1, pero no por el resumen FIA. Mantener el estado vigente del artefacto; registrar la diferencia de alcance entre visita y parada de servicio. |
| Ricciardo, Arabia Saudita 2022, candidato v.36 | La crónica de carrera de la FIA sitúa el incidente en v.37 y dice que se detuvo justo antes de la entrada al pit lane. | No contar como visita confirmada al pit lane. La vuelta y ubicación difieren del candidato de telemetría; dejarlo sin resolver hasta revisar la fila fuente y la semántica del campo. |
| Bottas, Arabia Saudita 2022, v.36 | La nota oficial de F1 informa que fue llamado a boxes en la vuelta 36 por un problema de motor/refrigeración. El resumen de paradas FIA no lo registra como parada de servicio. | Evidencia de visita a boxes en la crónica oficial; el resumen FIA no confirma una parada de servicio. No alterar el dataset en esta revisión. |
| Alonso, Emilia-Romaña 2022, v.6 | La nota oficial de F1 confirma que Alpine retiró el auto tras daños, pero no especifica que hubiera entrado al pit lane. La lista de paradas no incluye a Alonso. | Mantener `unknown`; el retiro no demuestra una visita a boxes. |

Fuentes consultadas:

- FIA, [informe de carrera de Arabia Saudita 2022](https://www.fia.com/news/f1-verstappen-beats-leclerc-thrilling-duel-saudi-arabia): Ricciardo se detuvo justo antes de la entrada al pit lane.
- F1, [reporte del GP de Baréin 2022](https://www.formula1.com/en/latest/article/leclerc-leads-1-2-for-ferrari-in-bahrain-gp-as-both-red-bulls-suffer-late.3GyOQOvMIMA8FIH17cXOoT): Verstappen regresó a boxes y abandonó en v.54.
- F1, [reporte sobre el abandono de Bottas en Arabia Saudita](https://www.formula1.com/en/latest/article/untitled-entry-2022-03-29-at-13-40-53.45AGWvsEQoZ2lWepCqGAqA.45AGWvsEQoZ2lWepCqGAqA): llamado a boxes en v.36.
- F1, [informe de equipos de Emilia-Romaña 2022](https://www.formula1.com/en/latest/article/what-the-teams-said-race-day-in-emilia-romagna-2022.4jX87kGINJ9o0mWVGbi9re.4jX87kGINJ9o0mWVGbi9re): retiro de Alonso por daños, sin ubicación explícita del auto.
- [Resumen FIA de paradas de Emilia-Romaña 2022](https://www.formula1.com/en/results/2022/races/1109/emilia-romagna/pit-stop-summary): no muestra parada de Alonso.

## Eventos en la última vuelta observada

Los 59 eventos restantes no tienen una fuente individual que ubique al auto en la calle de boxes. Se mantienen como desconocidos. No se convierten en negativos porque la ausencia de `PitOutTime` puede significar que acabó la telemetría, y no se convierten en positivos solo por coincidir con un retiro o la última vuelta disponible.

La siguiente revisión útil es inspeccionar cada fila fuente alrededor del evento y verificar si `PitInTime` es original o generado, su calidad (`IsAccurate`), el estado de `Stint` y si existe una narración oficial que indique explícitamente “pit lane”, “pits” o que el auto se detuvo en pista. Las filas que sigan sin prueba concluyente deben conservarse como desconocidas.

## Efecto sobre etiquetas

Este documento no modifica `etiquetado_pit_entry_v2`, sus eventos ni sus etiquetas. El artefacto mantiene 4.820 etiquetas positivas, 41.361 negativas y 1.816 desconocidas. La revisión confirma que hace falta una decisión de política antes de usar estos casos para entrenamiento: si la fuente normativa es exclusivamente el resumen de paradas FIA, una visita para retiro que no figure como parada de servicio no queda confirmada por ese documento; si se aceptan crónicas oficiales de F1, las visitas descritas allí aportan evidencia adicional.
