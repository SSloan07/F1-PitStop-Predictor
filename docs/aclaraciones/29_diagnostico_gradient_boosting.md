# Diagnóstico con gradient boosting: qué variables explican la entrada a boxes

**Estado:** completado en validación 2024, una semilla, determinista. La prueba 2025 no se cargó. Es un diagnóstico de variables, no un modelo candidato final.

## Pregunta

Ningún cambio de arquitectura LSTM (capacidad, dropout, profundidad, scheduler, `pos_weight`, quitar contexto estático) movió el F2 más que la variación entre semillas. Antes de seguir ajustando la red, se comprobó si el límite está en la información de entrada: ¿un modelo sin secuencia sobre las mismas muestras rinde igual?, ¿qué variables nuevas de estrategia, calculables con los datos que ya tenemos, aportan señal?

## Protocolo

- Mismas muestras supervisadas elegibles que la LSTM: 46.162 de train (2022–2023) y 25.635 de validación (2024; 9,0 % positivas). Mismas etiquetas v3.
- LightGBM 4.7 con parámetros fijos (`learning_rate=0.03`, `num_leaves=31`, `min_child_samples=50`, semilla 42, determinista). El número de árboles se eligió con validación cruzada agrupada por carrera **dentro de train** (5 particiones, parada temprana por AP). La validación 2024 solo se usa para el reporte y para el umbral F2 en la misma cuadrícula de la LSTM, igual que en los experimentos anteriores.
- Las variables base resumen la ventana causal de la LSTM: último valor visible y media de la ventana de cada numérica, última categoría dinámica e IDs estáticos.
- Las variables de estrategia se reconstruyen de la matriz de vueltas usando únicamente filas y eventos con tiempo `<= decision_time_seconds`. Un test (`tests/test_diagnostico_gbm.py`) verifica que alterar vueltas y paradas posteriores no cambia ningún valor.

| Variable nueva | Definición causal |
|---|---|
| `race_fraction`, `laps_remaining` | Vueltas completadas sobre la distancia programada (`TotalLaps` anunciado antes de la salida en la caché FastF1). |
| `stops_done`, `laps_since_last_stop` | `PitInTime` propios anteriores a la decisión. |
| `dry_compounds_used`, `two_dry_compounds_used`, `used_wet_tyres` | Compuestos de las vueltas propias ya completadas (regla de dos compuestos en seco). |
| `tyre_life_vs_typical_stint` | Vida del neumático actual dividida por la mediana de stints terminados en parada para ese compuesto en train (SOFT 14, MEDIUM 17, HARD 21,5 vueltas). |
| `pace_vs_stint_best`, `pace_slope_last5` | Última vuelta limpia (verde, sin entrada/salida de boxes) menos la mejor del stint actual; pendiente de las últimas cinco. |
| `field_pit_entries_last2laps`, `field_share_stopped` | Entradas a boxes de rivales en las últimas ~2 vueltas; fracción del pelotón que ya paró. |
| `gap_ahead_s`, `gap_behind_s`, `ahead/behind_pitted_last2laps` | Coche de adelante identificado en la vuelta N−1 y el de atrás en N−2 (ambos ya cruzaron la línea); diferencia de tiempos y si paró recientemente. |

Reproducción (CPU, unos 3 minutos; rechaza la carpeta si existe):

```bash
python -m pip install -r requirements-diagnostico.txt
python scripts/diagnostico_gbm.py --output docs/aclaraciones/evidencia_gbm_v1
```

## Resultados en validación 2024

| Entradas | Variables | AP | F2 | Precisión | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM32 referencia (Colab, `pos_weight=2`) | — | 0,2234 | 0,4622 | 18,2 % | 75,2 % | 7.822 | 575 |
| GBM equivalente a la LSTM | 49 | 0,2284 | 0,4582 | 19,2 % | 70,0 % | 6.807 | 694 |
| GBM solo telemetría | 32 | 0,2691 | 0,4776 | 21,7 % | 68,3 % | 5.700 | 735 |
| **GBM telemetría + estrategia** | 48 | **0,3333** | **0,5069** | 21,6 % | 76,6 % | 6.456 | 542 |
| GBM todo | 65 | 0,3409 | 0,5093 | 28,6 % | 63,3 % | 3.658 | 850 |

Lectura:

1. **Con la misma información, el GBM empata con la LSTM** (AP 0,228 frente a 0,223). La secuencia, tal como se usa hoy, no aporta más que un resumen de la última vuelta y la media de la ventana.
2. **Quitar el contexto estático mejora al GBM** (AP 0,228 → 0,269). Con IDs de piloto, equipo y circuito, el AP de train sube a 0,76: el modelo memoriza carreras de train que no se repiten en 2024.
3. **Las variables de estrategia son el salto grande:** AP +0,064 (+24 %) y F2 +0,029 sobre solo telemetría, con 193 falsos negativos menos. Es el primer cambio del proyecto claramente mayor que el ruido entre semillas (≈0,017 de F2).
4. Volver a añadir contexto estático y clima sobre la estrategia apenas suma AP (+0,008) y cambia el punto de operación hacia más precisión y menos recall.

## Qué variables influyen más

Importancia por permutación en validación (caída de AP al barajar la variable) en el modelo telemetría + estrategia. También se publican la ganancia de LightGBM y el SHAP medio en `evidencia_gbm_v1/importancia_variables.csv`.

| # | Variable | Caída de AP | Grupo |
|---:|---|---:|---|
| 1 | `tyre_life_vs_typical_stint` | 0,080 | neumático y stint (nueva) |
| 2 | `laps_since_last_stop` | 0,077 | neumático y stint (nueva) |
| 3 | `pace_vs_stint_best` | 0,055 | neumático y stint (nueva) |
| 4 | `field_pit_entries_last2laps` | 0,033 | rivales (nueva) |
| 5 | `laps_remaining` | 0,025 | progreso (nueva) |
| 6 | `gap_ahead_s` | 0,019 | rivales (nueva) |
| 7 | `race_fraction` | 0,014 | progreso (nueva) |
| 8 | `Compound_asof` (último) | 0,012 | neumático |
| 9 | `stops_done` | 0,010 | neumático y stint (nueva) |
| 10 | `Sector2Time` (último) | 0,008 | ritmo |

Por grupos (barajando todas las columnas del grupo a la vez):

| Grupo | Caída de AP |
|---|---:|
| Neumático y stint | 0,218 |
| Progreso de carrera | 0,078 |
| Posición y rivales | 0,053 |
| Ritmo (tiempos de vuelta y sectores) | 0,030 |
| Telemetría del coche (velocidad, RPM, acelerador, freno) | 0,024 |
| Clima (solo lluvia) | 0,002 |
| Estado de pista | ≈0 |

Señal univariante en validación (tasa de positivos):

- `stops_done`: 15,3 % sin paradas, 7,2 % con una, 3,7 % con dos.
- `two_dry_compounds_used`: 14,3 % si aún no usó dos compuestos de seco, 5,1 % si ya los usó.
- `field_pit_entries_last2laps`: 6,3 % si nadie entró, 16,1 % con tres o más entradas de rivales.
- `ahead_pitted_last2laps`: 8,6 % frente a 14,0 % si el coche de adelante acaba de parar.

**Conclusión:** lo que predice la parada es el **estado del stint relativo a lo normal** (cuánto lleva el neumático frente a un stint típico de ese compuesto, cuánto se ha degradado el ritmo dentro del stint, cuántas paradas faltan) y **lo que hace el pelotón**. La telemetría agregada del coche aporta poco por sí sola. `TyreLife` crudo ya estaba en la LSTM, pero sin normalizar por compuesto ni por el historial de paradas.

## Límites

- Una semilla y una temporada de validación, que también fija el umbral; son métricas de desarrollo.
- El F2 por muestra (0,51) sigue debajo de la meta provisional (F2 ≥ 0,60, precisión ≥ 30 %, recall ≥ 80 %).
- Bajo bandera no verde (987 muestras de validación), el recall cae a 43,6 % y el F2 a 0,40. Las paradas oportunistas por Safety Car siguen siendo difíciles de anticipar.
- Las variables nuevas se calcularon fuera del contrato causal de `muestras_causales_v1_3` (máscaras, latencias por fuente). Antes de usarlas en la LSTM deben incorporarse a ese constructor con su política de disponibilidad.

## Siguientes pasos

1. **Hecho:** las siete variables principales, más `stops_done` y `two_dry_compounds_used`, se incorporaron por posición de ventana con la regla causal y las máscaras del contrato ([especificación 20](../specs/20_experimento_lstm_estrategia.md)).
2. **Hecho (local, preliminar):** la pareja LSTM32 solo telemetría frente a telemetría + estrategia con tres semillas mejora F2 en las tres (media +0,044) y AP de 0,210 a 0,313. Pendiente la corrida oficial en Colab.
3. Mantener el GBM telemetría + estrategia como línea base fuerte: si la LSTM no lo supera, la secuencia no justifica su costo.
