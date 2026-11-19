# Segunda ronda de variables de estrategia: ablación en train y confirmación 2024

**Estado:** diagnóstico con LightGBM completado. La selección se hizo solo con train (2022–2023); la validación 2024 se usó una vez para confirmar. La prueba 2025 no se cargó. Las variables aún no están en ningún paquete de la LSTM.

## Pregunta

Tras la [primera ronda](29_diagnostico_gradient_boosting.md), ¿qué otras variables de estrategia, calculables con los datos que ya tenemos, añaden señal sobre el mejor conjunto de LightGBM (telemetría + estrategia, AP 0,333 en 2024)?

## Variables candidatas

Implementadas en `src/f1_pitstop/strategy_features_r2.py` con la misma regla causal: solo vueltas terminadas y entradas a boxes con tiempo `<= decision_time_seconds`. `tests/test_diagnostico_gbm.py` comprueba que alterar vueltas, paradas y lluvia posteriores no cambia ningún valor.

| Grupo | Variables | Idea |
|---|---|---|
| Llegar al final | `laps_short_to_finish` | Vueltas restantes menos la vida típica que le queda al neumático actual. |
| Compañero | `teammate_pitted_last2laps`, `teammate_stops_minus_own` | Paradas dobles y desfase de estrategia dentro del equipo (sin usar la identidad del equipo como entrada). |
| Ventana de parada | `pit_loss_est_s`, `free_stop_margin_s`, `cars_within_pit_loss_behind` | Pérdida estimada por parar (paradas previas en verde de la misma carrera; si hay menos de dos, mediana del mismo GP en otras carreras de train) y cuántos coches pasarían. |
| Undercut | `nearest_behind_pitted_last2laps`, `behind_faster_by_s` | El coche más cercano detrás acaba de parar y/o gira más rápido. |
| Safety Car | `sc_vsc_last_lap`, `cheap_stop_index` | SC/VSC en la última vuelta y su cruce con la edad relativa del neumático. |
| Lluvia | `rain_on_slicks`, `dry_on_wets` | Neumático incompatible con la pista. |
| Pelotón | `share_field_fresher_tyres` | Fracción de rivales cuya última parada fue posterior a la propia. |

## Protocolo

- Base: el conjunto LightGBM de la primera ronda (telemetría + estrategia, 48 variables).
- Validación cruzada de 5 particiones por carrera (`GroupKFold`) dentro de train, con parada temprana en el 20 % de las carreras de cada partición de entrenamiento. Todas las variantes usan las mismas particiones, así que las diferencias son pareadas.
- Variantes: base, base + cada grupo, base + todos y base + todos − cada grupo.
- **Regla fijada antes de ver resultados:** un grupo se conserva si quitarlo de "base + todos" baja el AP fuera de partición más de 0,002 en promedio y en al menos 4 de 5 particiones.
- Confirmación única en 2024 de base, base + seleccionados y base + todos.

```bash
python scripts/diagnostico_gbm_ronda2.py --output docs/aclaraciones/evidencia_gbm_ronda2_v1
```

## Resultados en train (validación cruzada)

| Variante | AP fuera de partición | F2 fuera de partición |
|---|---:|---:|
| Base | 0,3216 | 0,5040 |
| Base + todos los grupos | 0,3433 | 0,5075 |

El bloque completo mejora el AP en 4 de 5 particiones (Δ media por partición +0,017; +0,055, −0,002, +0,013, +0,008, +0,012).

Por grupo (ΔAP media por partición; entre paréntesis, particiones con mejora):

| Grupo | Añadido a la base | Aporte sobre los demás (quitarlo de "todos") | ¿Pasa la regla? |
|---|---:|---:|---|
| Compañero | +0,009 (3/5) | +0,006 (3/5) | No |
| Pelotón | +0,004 (3/5) | +0,009 (3/5) | No |
| Ventana de parada | +0,009 (4/5) | −0,002 (2/5) | No |
| Undercut | +0,010 (2/5) | −0,003 (2/5) | No |
| Lluvia | +0,006 (2/5) | −0,003 (2/5) | No |
| Safety Car | 0,000 (2/5) | −0,003 (3/5) | No |
| Llegar al final | 0,000 (2/5) | −0,000 (2/5) | No |

**Ningún grupo pasa la regla por sí solo.** La varianza entre particiones es grande (el AP de la base va de 0,25 a 0,44 según las carreras) y los grupos se solapan: ventana de parada, undercut, lluvia y Safety Car mejoran algo al añadirlos a la base, pero no aportan nada cuando ya están los demás. Compañero y pelotón son los únicos con aporte propio positivo, aunque solo en 3 de 5 particiones.

## Confirmación en validación 2024

Como la regla no seleccionó grupos, la confirmación compara base y base + todos (ambas previstas en el protocolo):

| Validación 2024 | AP | F2 | F1 | Precisión | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base (48 variables) | 0,3333 | 0,5069 | 0,3363 | 21,5 % | 76,6 % | 6.456 | 542 |
| Base + todos (61 variables) | **0,3693** | **0,5344** | **0,4014** | **28,4 %** | 68,6 % | **4.010** | 727 |

En 2024 el bloque completo sube AP +0,036 y F2 +0,028, con 2.446 falsas alertas menos y 185 falsos negativos más. Es coherente con la validación cruzada en train.

**Importancia en 2024 (permutación, modelo base + todos).** `share_field_fresher_tyres` es la **segunda variable más importante de todo el modelo** (caída de AP 0,066), solo detrás de `tyre_life_vs_typical_stint` (0,071) y por delante de `pace_vs_stint_best` (0,051) y `laps_since_last_stop` (0,035). Después aparecen, mucho más abajo, `teammate_stops_minus_own` (0,006) y `behind_faster_by_s` (0,006). Las demás variables nuevas aportan menos de 0,003 cada una.

## Conclusiones

1. La segunda ronda mejora como bloque (train: 4/5 particiones; 2024: AP 0,333 → 0,369, F2 0,507 → 0,534), pero con la regla por grupo fijada de antemano no se puede atribuir la mejora a un grupo concreto.
2. La señal más clara es **cuántos rivales llevan neumáticos más nuevos**: la presión del pelotón que ya paró. Le sigue el desfase de estrategia con el compañero.
3. Las variables pensadas para Safety Car, lluvia y llegar al final no ayudan de forma medible. Las situaciones que buscan son poco frecuentes (por ejemplo, 987 decisiones bajo bandera no verde en 2024), así que tienen pocos ejemplos.
4. Seguimos debajo de la meta provisional (F2 ≥ 0,60, precisión ≥ 30 %, recall ≥ 80 %), aunque la precisión ya roza el 30 %.

## Siguientes pasos

- **Hecho (local, preliminar):** el bloque completo se llevó a la LSTM ([especificación 21](../specs/21_experimento_lstm_estrategia_r2.md)). Con tres semillas no repite la mejora de LightGBM: ΔF2 medio +0,006 (2/3 semillas) y ΔAP −0,018 (mejora en 1/3). Pendiente la corrida oficial en Colab.
- Pendiente: un subconjunto reducido (`share_field_fresher_tyres` y las de compañero) elegido con validación cruzada en train, más regularización en la LSTM, o el ensamble LightGBM + LSTM.
