# Cierre de la revisión de eventos `PitInTime` sin resolver

## Alcance

Esta revisión completa la clasificación de las entradas `PitInTime` candidatas en las carreras de 2022 y 2023. La pregunta auditada es si el coche visitó la calle de boxes en el evento indicado; no se infiere que cambiara neumáticos ni que la visita fuera una parada de servicio.

Se conserva como evidencia primaria el análisis de vueltas y los resúmenes de la FIA cuando registran el evento. Para entradas de retiro que no aparecen en un resumen de paradas, se usaron informes contemporáneos de Formula 1 o del equipo, contrastados con el evento local de FastF1 y, cuando estaba disponible, la clasificación FIA. Las diferencias entre vuelta de carrera y vuelta completada por el piloto se anotan explícitamente.

## Casos resueltos en esta iteración

| Evento | Resolución | Evidencia |
|---|---|---|
| Arabia Saudita 2022 — Daniel Ricciardo, vuelta FastF1 36 | Entrada confirmada en la entrada del pit lane al detenerse. El resumen FIA lista su parada de servicio de la vuelta 8; este caso es la visita posterior de retiro. | [Informe de carrera de Formula 1](https://www.formula1.com/en/latest/article.verstappen-beats-leclerc-by-just-0-5s-in-epic-saudi-arabian-grand-prix.oTicpPnf2hMbdjTzCyROM.html) |
| Miami 2022 — Pierre Gasly, vuelta FastF1 45 | Entrada confirmada: el piloto describe que regresaba a boxes al producirse el contacto que precedió su retiro. | [Video/entrevista de Formula 1](https://www.formula1.com/en/video/pierre-gasly-extremely-disappointed-with-end-to-miami-gp.1732296948953501243) |
| Miami 2022 — Kevin Magnussen, vuelta FastF1 57 | Entrada confirmada: Haas informa que entró a boxes en la última vuelta por daños. El equipo reporta 56 vueltas completadas; se conserva el contador local de FastF1 para el evento. | [Informe de Haas](https://www.haasf1team.com/news/miami-grand-prix-race-recap) |
| Francia 2022 — Kevin Magnussen, vuelta 38 | Entrada confirmada: Haas indica que entró al final de la vuelta 38 para retirar el coche. | [Informe de Haas](https://www.haasf1team.com/news/french-grand-prix-race-recap-2) |
| Francia 2022 — Nicholas Latifi, vuelta FastF1 41 | Entrada confirmada tras el choque: Formula 1 dice que Latifi siguió a Magnussen a boxes un par de vueltas después. La fuente expresa el intervalo respecto al incidente, no su contador individual. | [Informe de Formula 1](https://www.formula1.com/en/latest/article/verstappen-wins-the-2022-french-grand-prix-as-pole-sitter-leclerc-crashes.11eNOAzyvZfPOnRnY3AYuz.11eNOAzyvZfPOnRnY3AYuz) |
| Francia 2022 — Yuki Tsunoda, vuelta FastF1 18 | Entrada confirmada: RaceFans especifica que fue llamado a boxes para retirarse en la vuelta 18; Formula 1 confirma el abandono temprano por daños después del contacto con Ocon. | [RaceFans](https://www.racefans.net/2022/07/26/2022-french-grand-prix-driver-ratings/), [informe de Formula 1](https://www.formula1.com/en/latest/article/verstappen-wins-the-2022-french-grand-prix-as-pole-sitter-leclerc-crashes.11eNOAzyvZfPOnRnY3AYuz.11eNOAzyvZfPOnRnY3AYuz) |
| Baréin 2023 — Esteban Ocon, vuelta FastF1 41 | Entrada confirmada: Formula 1 dice que entró al pit lane y fue retirado en la vuelta 43 de carrera. Ocon llevaba dos vueltas perdidas; la fila FastF1 usa la vuelta completada por el coche (41). | [Informe de Formula 1](https://www.formula1.com/en/latest/article/everything-went-wrong-today-says-ocon-after-being-penalised-three-times-and.1Obkye7mOTceYVpzCX9Bp6) |
| Mónaco 2023 — Kevin Magnussen, vuelta FastF1 71 | Entrada confirmada: Formula 1 informa que volvió a boxes antes de retirarse. La fuente no especifica el contador individual de vuelta; se mantiene el tiempo y la vuelta observados localmente. | [Informe de Formula 1](https://www.formula1.com/en/latest/article/magnussen-says-haas-tried-everything-in-monaco-as-hulkenberg-hopes-struggles.6iiHcCX7kdQPYvfJ3IVzwP) |
| Canadá 2023 — George Russell, vuelta FastF1 54 | Entrada confirmada por cruce: Formula 1 informa que en la vuelta 55 de carrera Mercedes le ordenó entrar a boxes y retirar el coche. Russell iba una vuelta detrás; FastF1 marca la vuelta completada 54. | [Informe de carrera de Formula 1](https://www.formula1.com/en/latest/article/verstappen-wins-canadian-gp-to-claim-red-bulls-100th-victory-and-equal.7A6g45iAe5jqGGUc5k153Y) |
| Ciudad de México 2023 — Logan Sargeant, vuelta FastF1 71 | Entrada confirmada: Williams dice que decidió llevarlo a boxes en la última vuelta en vez de tomar la bandera para proteger la unidad de potencia. | [Informe de Williams](https://www.williamsf1.com/articles/47ee7941-cf34-4fef-9d09-d6ecd72f085d/report-2023-mexico-city-grand-prix) |

Ninguno de estos casos se convirtió en negativo por ausencia en el resumen de paradas: cuando el registro no demuestra una parada de servicio, los informes narrativos de retiro se usaron solo para corroborar la entrada a boxes, de acuerdo con el objetivo actual de visita a pit lane. No se infiere que esas visitas fueran cambios de neumáticos.

## Resultado reproducible

La corrida nueva está en `EDA/dataset/datos_carreras_completos_2022_2025/etiquetado_pit_entry_v19/` y su auditoría acumulativa en `EDA/dataset/datos_carreras_completos_2022_2025/etiquetado_candidate_v17/casos_revision_usuario.csv`.

- 1.779 eventos: 1.779 clasificados y 0 sin resolver.
- 76 confirmados mediante registros FIA directos.
- 16 confirmados con evidencia cruzada contemporánea (F1/equipo/FastF1; compuestos no inferidos).
- Etiquetas de horizonte: 4.990 positivas, 41.361 negativas y 1.646 desconocidas. Las desconocidas de las etiquetas también incluyen horizontes incompletos o muestras sin datos suficientes; no equivalen a eventos `PitInTime` pendientes.
- La corrida sigue marcada como candidata; no se entrenó ningún modelo y no se incluyeron temporadas 2024–2025.

Las versiones anteriores permanecen intactas. El manifiesto de v19 fija las rutas y hashes de la matriz de entrada y del CSV de auditoría usados para construir el resultado.

## Diagnóstico de etiquetas desconocidas

Se comparó cada muestra desconocida con la última vuelta observada para ese piloto en esa carrera. Las 1.646 desconocidas están al final de su secuencia: 1.567 tienen horizonte incompleto (732 en la última vuelta observada, 779 a una vuelta del final y 56 a dos vueltas) y 79 son muestras actuales inválidas en la última vuelta. No se encontraron desconocidas a más de dos vueltas del final observado.

Esto apunta a censura al final de carrera/abandono y a muestras FastF1 inválidas, no a huecos internos detectados por esta comprobación. Las desconocidas deben conservarse como `pd.NA`; no se deben convertir en negativas, porque no se observa el horizonte completo de tres vueltas. Este diagnóstico no valida aún la calidad de cada vuelta ni sustituye el análisis posterior de filas y variables.
