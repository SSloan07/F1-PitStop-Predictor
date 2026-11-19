# Tercera ronda: variables sobre el número de paradas

**Estado:** diagnóstico con LightGBM completado. La selección se hizo con train y la confirmación con 2024, una vez. La prueba 2025 no se cargó.

## Pregunta

El [análisis de errores](32_analisis_errores.md) mostró que casi la mitad de las falsas alertas son alarmas puras: el piloto no vuelve a parar, o para 11 vueltas o más después. El modelo espera una parada por el desgaste y el piloto estira el stint o hace una parada menos. ¿Ayudan variables que describan el plan de paradas?

## Variables

Implementadas en `src/f1_pitstop/strategy_features_r3.py`. En la carrera actual solo usan vueltas terminadas y entradas a boxes con tiempo `<= decision_time_seconds`; los valores de referencia salen solo de train. `tests/test_diagnostico_gbm.py` comprueba que alterar datos futuros no cambia ningún valor.

| Variable | Definición |
|---|---|
| `circuit_typical_stops` | Mediana de paradas de los pilotos clasificados (≥ 90 % de las vueltas) en otras carreras de train del mismo GP; si no hay, mediana de todas las de train. |
| `expected_stops_remaining` | `circuit_typical_stops` menos las paradas propias hasta ahora (mínimo 0). |
| `stops_vs_field_mean` | Paradas propias menos la media de paradas de los rivales hasta ahora. |
| `field_share_done_typical` | Fracción de rivales que ya hicieron el número típico de paradas. |
| `laps_beyond_long_stint` | Vueltas restantes menos lo que le queda al neumático actual si se estira a un stint largo (percentil 90 de train: SOFT 26, MEDIUM 31, HARD 42 vueltas). Positivo: hace falta otra parada. |

## Protocolo

- Base: el mejor conjunto LightGBM actual (telemetría + estrategia de rondas 1 y 2, 61 variables). Candidata: base + estas 5 variables.
- Validación cruzada de 5 particiones por carrera en train, igual que en la [ronda 2](30_segunda_ronda_variables_estrategia.md).
- **Regla fijada antes de ver resultados:** se adoptan si suben el AP fuera de partición más de 0,002 en promedio y en al menos 4 de 5 particiones.
- Confirmación única en 2024, contando además las alarmas puras.

```bash
python scripts/diagnostico_gbm_ronda3.py --output docs/aclaraciones/evidencia_gbm_ronda3_v1
```

El entorno se recreó (pandas 2.3, LightGBM 4.7). La base reproduce exactamente los valores anteriores (AP 0,3693, F2 0,5344 en 2024).

## Resultados

**Train (validación cruzada):** AP fuera de partición 0,3433 → 0,3576 y F2 0,5075 → 0,5089. ΔAP por partición: +0,007, +0,027, −0,028, +0,027 y +0,004 (media +0,0076; 4/5 positivas). **Se cumple la regla y las variables se adoptan**, aunque una partición empeora claramente.

**Confirmación 2024:**

| Validación 2024 | AP | F2 | Precisión | Recall | FP | FN | Alarmas puras (% de FP) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base (61 variables) | 0,3693 | 0,5344 | 28,4 % | 68,6 % | 4.010 | 727 | 1.765 (44,0 %) |
| Base + paradas (66) | **0,3783** | 0,5327 | 26,6 % | 71,0 % | 4.526 | 671 | 2.033 (44,9 %) |

- **El AP sube +0,009:** el modelo ordena mejor las muestras, en línea con lo visto en train.
- **El F2 no cambia** (−0,002). El umbral elegido baja de 0,15 a 0,14: el modelo cubre 56 positivos más con 516 falsas alertas más.
- **Las alarmas puras no bajan:** siguen siendo el 44–45 % de las falsas alertas. El objetivo que motivó la ronda no se consigue.

**Importancia en 2024 (permutación):** `stops_vs_field_mean` queda **tercera de todo el modelo** (caída de AP 0,048), junto a `share_field_fresher_tyres` (0,052), `tyre_life_vs_typical_stint` (0,049) y `pace_vs_stint_best` (0,047). `laps_beyond_long_stint` aporta poco (0,008). `expected_stops_remaining`, `field_share_done_typical` y `circuit_typical_stops` no aportan; las dos últimas tienen importancia ligeramente negativa.

## Conclusiones

1. La comparación con el pelotón vuelve a ser la señal útil: estar por delante o por detrás en número de paradas (`stops_vs_field_mean`), como antes la fracción de rivales con neumáticos más nuevos.
2. El número típico de paradas por circuito no ayuda. Con dos temporadas de train hay pocas carreras por GP y la estrategia varía mucho entre años.
3. **La mejora es de ordenación (AP), no de decisión (F2).** Las alarmas puras siguen ahí: con la información causal disponible, el modelo no distingue cuándo un equipo decidirá estirar el stint o hacer una parada menos.
4. Mejor conjunto LightGBM actual: 66 variables, AP 0,378 y F2 0,533 en 2024 (validación ya muy usada; métricas de desarrollo).

## Siguientes pasos

- Las rondas de variables están dando rendimientos decrecientes en F2. Lo siguiente con más sentido es ajustar los hiperparámetros de LightGBM con validación cruzada en train y después congelar el modelo para evaluarlo una sola vez en 2025.
- Para el informe: reportar el F2 por muestra junto con la cobertura por tipo de parada y el F2 con tolerancia ([análisis de errores](32_analisis_errores.md)), porque el F2 por muestra no refleja la mejora en AP.
