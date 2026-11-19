# Experimento pareado con tres semillas: LSTM32 solo telemetría frente a telemetría + estrategia

**Estado:** preparado para Colab CPU; ejecución local preliminar completada (tres semillas). El test 2025 permanece reservado.

## Pregunta

El [diagnóstico con gradient boosting](../aclaraciones/29_diagnostico_gradient_boosting.md) mostró que el estado de la estrategia (edad del stint relativa a lo típico, paradas, pérdida de ritmo, actividad de boxes de los rivales, progreso de la carrera) explica la entrada a boxes mucho mejor que la telemetría del coche. ¿La LSTM aprovecha esa información si se le entrega vuelta a vuelta?

## Entradas

`EDA/dataset/muestras_preprocesadas_v1_3_telemetria_estrategia_v1/` extiende el paquete solo telemetría (`muestras_preprocesadas_v1_3_telemetria_v1`, [especificación 19](19_experimento_lstm_solo_telemetria.md)) con nueve numéricas por posición de la ventana:

| Variable | Significado |
|---|---|
| `tyre_life_vs_typical_stint` | Vida del neumático / mediana de stints terminados en parada para ese compuesto (train). |
| `laps_since_last_stop` | Vueltas desde la última entrada a boxes propia. |
| `pace_vs_stint_best` | Última vuelta limpia menos la mejor del stint actual (s). |
| `field_pit_entries_last2laps` | Entradas a boxes de rivales en las últimas ~2 vueltas. |
| `laps_remaining`, `race_fraction` | Distancia programada restante y fracción completada. |
| `gap_ahead_s` | Diferencia de tiempo con el coche de adelante en la última línea cruzada. |
| `stops_done`, `two_dry_compounds_used` | Paradas hechas y si ya usó dos compuestos de seco. |

**Regla causal por posición.** Para una muestra que predice la vuelta N, la posición de la ventana que muestra la vuelta L ≤ N lleva el valor conocido al inicio de la vuelta L (su propio `decision_time_seconds`). La última posición tiene exactamente el estado de la estrategia en la decisión. Los cálculos usan solo filas y entradas a boxes con tiempo no posterior a ese instante (`src/f1_pitstop/strategy_features.py`, verificado por `tests/test_diagnostico_gbm.py`).

**Máscaras.** Padding → 1, fila de vuelta ausente → 2, posición con instante posterior a la decisión → 3, valor no calculable (p. ej., sin vuelta limpia en el stint o sin coche adelante) → 4. Los valores enmascarados se guardan como cero técnico.

**Ajuste.** Medias y desviaciones con celdas presentes de ventanas train elegibles (misma regla del preprocesador). Las medianas de stint usan solo 2022–2023. 2025 no se carga. Las numéricas, categóricas, máscaras, etiquetas y orden previos se copian sin cambios; se comprobó que la última posición coincide con las variables del diagnóstico GBM (diferencia máxima ≈10⁻⁶ por redondeo float32).

```bash
python scripts/preparar_entradas_estrategia.py --config configs/entradas_estrategia_v1.json
```

## Diseño ortogonal

| Control | Referencia | Candidata |
|---|---|---|
| Entradas | `..._telemetria_v1` (15 numéricas) | `..._telemetria_estrategia_v1` (24 numéricas) |
| Arquitectura | LSTM32 de una capa, FFNN estática 16 (marcador constante), cabeza 16, dropout 0,2 | Igual |
| Pérdida / optimizador / tasa | BCE `pos_weight=2` / Adam / 0,001 fija | Igual |
| Selección | F2 a 0,5 para checkpoint; F2 en la cuadrícula congelada para el umbral | Igual |
| Semillas / entorno | 42, 43 y 44 / Colab CPU, cada pareja en la misma sesión | Igual |

El runner `scripts/entrenar_comparacion_telemetria.py` (generalizado para ambos experimentos de entradas) exige que solo cambie el paquete de entrada y que la candidata declare derivar de la referencia. Configuraciones: `configs/experimento_lstm32_estrategia_{ref,candidate}_cpu_seed{42,43,44}_v1.json`.

## Ejecución

1. `python scripts/empaquetar_estrategia.py --output runs/lstm_colab_estrategia_cpu_v1.zip` (el ZIP publicado y su SHA-256 están fijados en el notebook).
2. Abrir `notebooks/13_lstm32_estrategia_f2_colab_cpu.ipynb` en Colab, subir el ZIP y ejecutar todo. Entrena seis modelos.
3. Devolver los tres `lstm32_estrategia_f2_cpu_v1_seed{42,43,44}.zip` intactos a `runs/` y evaluar localmente con `scripts/evaluar_rnn.py` (`--data` del paquete correspondiente a cada hijo).

## Resultado local preliminar (no oficial)

Ejecución en CPU local con el mismo código; evaluación local con paridad exacta y cero decisiones distintas en los seis modelos. Evidencia: [`evidencia_lstm_estrategia_local_v1.json`](../aclaraciones/evidencia_lstm_estrategia_local_v1.json).

| Semilla | F2 telemetría → + estrategia | AP | Precisión | Recall | FP | Eventos cubiertos |
|---:|---|---|---|---|---|---|
| 42 | 0,4588 → 0,4812 (+0,022) | 0,212 → 0,286 | 19,3 → 22,6 % | 69,9 → 67,0 % | 6.763 → 5.315 | 614 → 618 |
| 43 | 0,4657 → 0,5184 (+0,053) | 0,208 → 0,324 | 19,0 → 24,5 % | 73,3 → 71,9 % | 7.255 → 5.135 | 663 → 654 |
| 44 | 0,4518 → 0,5089 (+0,057) | 0,209 → 0,330 | 18,7 → 25,2 % | 70,1 → 68,3 % | 7.077 → 4.701 | 637 → 638 |
| **Media** | **0,4588 → 0,5028 (+0,044)** | **0,210 → 0,313 (+0,104)** | **+5,1 pp** | **−2,0 pp** | **≈ −1.980** | **≈ igual** |

Lectura:

- **Es la primera mejora consistente del proyecto:** F2 y AP suben en las tres semillas, y la ganancia media de F2 (+0,044) supera con holgura la variación entre semillas (≈0,014 entre referencias).
- La ganancia viene de **menos falsas alertas** (≈1.980 menos en promedio, precisión +5 pp) con un recall ligeramente menor; la cobertura por evento queda prácticamente igual (+4, −9, +1 sobre 824).
- La LSTM con estrategia (AP 0,313, F2 0,503) queda a la par del LightGBM con las mismas variables (AP 0,333, F2 0,507): la secuencia todavía no añade sobre un resumen de la última vuelta.
- Las candidatas alcanzan su mejor época antes (5–15 frente a 13–22), señal de que aprenden rápido y conviene vigilar el sobreajuste.
- Sigue debajo de la meta provisional (F2 ≥ 0,60, precisión ≥ 30 %, recall ≥ 80 %).

La corrida oficial es la de Colab; los números locales pueden diferir ligeramente por entorno.
