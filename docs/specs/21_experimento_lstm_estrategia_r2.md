# Experimento pareado con tres semillas: LSTM32 estrategia ronda 1 frente a ronda 1 + bloque completo de ronda 2

**Estado:** preparado para Colab CPU; ejecución local preliminar completada (tres semillas). El test 2025 permanece reservado.

## Pregunta

La [segunda ronda de variables](../aclaraciones/30_segunda_ronda_variables_estrategia.md) mejoró a LightGBM como bloque (AP 0,333 → 0,369 y F2 0,507 → 0,534 en 2024; AP mejor en 4 de 5 particiones de validación cruzada en train). ¿La LSTM, que ya usa la estrategia de la ronda 1 ([especificación 20](20_experimento_lstm_estrategia.md)), también mejora con ese bloque?

## Entradas

`EDA/dataset/muestras_preprocesadas_v1_3_telemetria_estrategia_r2_v1/` extiende `..._telemetria_estrategia_v1` con las 13 variables de la ronda 2 por posición de la ventana (37 numéricas en total): `laps_short_to_finish`, `teammate_pitted_last2laps`, `teammate_stops_minus_own`, `pit_loss_est_s`, `free_stop_margin_s`, `cars_within_pit_loss_behind`, `nearest_behind_pitted_last2laps`, `behind_faster_by_s`, `sc_vsc_last_lap`, `cheap_stop_index`, `rain_on_slicks`, `dry_on_wets` y `share_field_fresher_tyres`. Se usa la misma regla causal por posición, las mismas máscaras y estadísticas ajustadas solo con train que en la especificación 20 (`strategy_round: 2` en `configs/entradas_estrategia_r2_v1.json`).

```bash
python scripts/preparar_entradas_estrategia.py --config configs/entradas_estrategia_r2_v1.json
```

## Diseño ortogonal

La referencia es exactamente la candidata de la especificación 20; la candidata solo cambia el paquete de entrada. Arquitectura LSTM32 (una capa, dropout 0,2), BCE `pos_weight=2`, Adam 0,001 y selección por F2. Semillas 42, 43 y 44, cada pareja en la misma sesión CPU. Configuraciones: `configs/experimento_lstm32_estrategia_r2_{ref,candidate}_cpu_seed{42,43,44}_v1.json`. El runner es el mismo (`scripts/entrenar_comparacion_telemetria.py`, experimento `lstm32_estrategia_r2_f2_v1`).

## Ejecución

1. `python scripts/empaquetar_estrategia.py --experiment r2` crea `runs/lstm_colab_estrategia_r2_cpu_v1.zip`. El SHA-256 está fijado en el notebook.
2. Ejecutar `notebooks/14_lstm32_estrategia_r2_f2_colab_cpu.ipynb` en Colab con ese ZIP.
3. Devolver los tres `lstm32_estrategia_r2_f2_cpu_v1_seed{42,43,44}.zip` a `runs/`.

## Resultado local preliminar (no oficial)

Evaluación local con paridad exacta y cero decisiones distintas en los seis modelos. Evidencia: [`evidencia_lstm_estrategia_r2_local_v1.json`](../aclaraciones/evidencia_lstm_estrategia_r2_local_v1.json).

| Semilla | F2 r1 → r1 + r2 | AP | Precisión | Recall | FP | Eventos cubiertos |
|---:|---|---|---|---|---|---|
| 42 | 0,4812 → 0,5047 | 0,286 → 0,281 | 22,6 → 21,8 % | 67,0 → 75,2 % | 5.315 → 6.247 | 618 → 694 |
| 43 | 0,5184 → 0,5197 | 0,324 → 0,336 | 24,5 → 25,1 % | 71,9 → 70,9 % | 5.135 → 4.895 | 654 → 633 |
| 44 | 0,5089 → 0,5012 | 0,330 → 0,269 | 25,2 → 22,4 % | 68,3 → 72,6 % | 4.701 → 5.821 | 638 → 682 |
| **Media** | **0,5028 → 0,5085 (+0,006)** | **0,313 → 0,295 (−0,018)** | **−1,0 pp** | **+3,8 pp** | **+604** | **+33** |

Lectura:

- **La LSTM no repite la mejora que el bloque dio a LightGBM.** El F2 sube en 2 de 3 semillas y apenas +0,006 de media, dentro del ruido. El AP baja en 2 de 3 (−0,018 de media), con una caída fuerte en la semilla 44.
- Lo que cambia es el punto de operación: en dos semillas el umbral elegido baja, el recall sube y se cubren más eventos (+76 y +44), a costa de más falsas alertas.
- Con 37 entradas, la LSTM32 alcanza su mejor época muy pronto (5–8) y los resultados varían más entre semillas que con la ronda 1. Esto apunta a sobreajuste o a dificultad para aprovechar variables escasas o con colas largas (`pit_loss_est_s`, `free_stop_margin_s`), que LightGBM maneja con cortes.
- **Por ahora, la mejor LSTM sigue siendo la de la ronda 1** (especificación 20: F2 medio 0,503, AP 0,313). Queda por debajo del LightGBM con ronda 1 + ronda 2 (AP 0,369, F2 0,534).

## Implicaciones

- Para que la LSTM aproveche el bloque, las opciones con más sentido son: más regularización (dropout de entrada o weight decay), una transformación robusta de las variables con colas largas, o solo el subconjunto con aporte propio (`share_field_fresher_tyres` y las de compañero), elegido con validación cruzada en train.
- La alternativa directa es combinar las probabilidades de LightGBM (ronda 1 + ronda 2) y la LSTM de la ronda 1, o adoptar LightGBM como modelo principal si la LSTM no lo supera.
