# Resultado final en la temporada de prueba 2025

**Estado:** evaluación única ejecutada el 2026-11-19 con el modelo congelado (`modelos/lightgbm_final_v1/`, manifiesto `e23908f3…`) y el protocolo pre-registrado en la [especificación 22](../specs/22_modelo_final_y_protocolo_test_2025.md). No se reajustó nada con datos de 2025. Los resultados se reportan completos, también los desfavorables.

```bash
python scripts/evaluar_modelo_final.py --split test --confirmar-evaluacion-unica
```

Artefactos: `docs/aclaraciones/evaluacion_final_2025/` (métricas, predicciones, eventos, F2 por carrera y manifiesto).

## Métrica principal

**F2 por muestra en 2025, al umbral congelado 0,14: 0,474.**

## Comparación con la validación 2024

| Métrica | Validación 2024 (desarrollo) | **Prueba 2025** | Diferencia |
|---|---:|---:|---:|
| F2 (principal) | 0,540 | **0,474** | −0,066 |
| AP | 0,387 | **0,332** | −0,055 |
| F1 | 0,399 | 0,341 | −0,058 |
| Precisión | 27,9 % | 23,3 % | −4,6 pp |
| Recall por muestra | 70,5 % | 64,0 % | −6,5 pp |
| Falsos positivos / negativos | 4.229 / 682 | 4.767 / 814 | — |
| Muestras elegibles (positivas) | 25.635 (9,0 %) | 25.741 (8,8 %) | — |
| Cobertura por evento | 650/824 (78,9 %) | 613/820 (74,8 %) | −4,1 pp |
| — en verde con stint normal | 86,1 % (592) | 82,9 % (543) | −3,2 pp |
| — SC/VSC con stint normal | 78,1 % (96) | 68,9 % (196) | −9,2 pp |
| — stint corto (< 50 % del típico) | 47,8 % (136) | 34,6 % (81) | −13,2 pp |
| F2 con tolerancia N+3/N+4 | 0,571 | 0,500 | −0,071 |
| F2 por carrera (mín. / mediana / máx.) | 0,19 / 0,55 / 0,80 | 0,16 / 0,48 / 0,74 | — |

Distribución de las falsas alertas en 2025 según la siguiente entrada real: N+3 8,7 %, N+4 7,9 %, N+5–6 14,0 %, N+7–10 21,9 %, N+11 o más 31,1 %, sin parada posterior 16,4 %.

## Lectura

1. **El rendimiento baja en 2025, como cabía esperar.** La validación 2024 se usó para muchas decisiones (variables, confirmaciones, umbral), así que sus métricas eran optimistas. Parte de la caída de F2 (−0,066) se debe a ese optimismo; la otra parte, a que 2025 fue distinta (punto 3).
2. **El núcleo del modelo se mantiene:** las paradas planificadas en verde con stint normal se cubren al 82,9 % (86,1 % en 2024), casi siempre con dos vueltas de antelación.
3. **2025 tuvo otra mezcla de paradas.** Las paradas bajo SC/VSC pasaron del 21 % al 31 % de los eventos (196 con stint normal frente a 96). Es justo el tipo de parada que peor se anticipa, y su cobertura también bajó (68,9 %).
4. **Las peores carreras de 2025** son Miami (F2 0,16), Australia (0,17), Mónaco (0,20) y Qatar (0,21). En Australia hubo lluvia (también en Gran Bretaña, quinta peor con 0,36), en Mónaco una regla de paradas obligatorias propia de 2025 y en Qatar un límite de vueltas por neumático: condiciones que no estaban en train. Las mejores son Baréin (0,74), Bélgica (0,62) y Austria (0,61). El protocolo prohíbe excluir carreras: el resultado principal las incluye todas.
5. **Las falsas alertas son sobre todo alarmas puras**, igual que en 2024: el 47 % tiene la siguiente parada a 11 vueltas o más, o ninguna. El F2 con tolerancia (0,500) apenas supera al estricto.

## Conclusión para el proyecto

- El modelo final (LightGBM con variables causales de telemetría y estrategia) obtiene en la temporada no vista **F2 0,474, AP 0,332, precisión 23,3 % y recall 64,0 %**, y anticipa el 74,8 % de las paradas (82,9 % de las planificadas en verde).
- Queda por debajo de la meta provisional (F2 ≥ 0,60, precisión ≥ 30 %, recall ≥ 80 %).
- Los límites principales son los identificados en desarrollo: paradas oportunistas o tempranas, y decisiones de estirar el stint que la información causal disponible no permite prever. Además, 2025 trajo reglas y condiciones de carrera que no estaban en train.
- Este es el resultado honesto del proyecto. Cualquier ajuste posterior hecho mirando 2025 ya no contaría como prueba independiente.
