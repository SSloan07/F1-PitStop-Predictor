# Comparación a posteriori de las LSTM en 2025

**Estado:** declarada y commiteada **antes** de evaluar las LSTM en 2025. Es una comparación a posteriori: el LightGBM final ya se evaluó en 2025 ([informe 35](../aclaraciones/35_resultado_final_test_2025.md)). No forma parte del protocolo original ([especificación 22](22_modelo_final_y_protocolo_test_2025.md)), y así debe citarse.

## Objetivo

Medir en la temporada no vista cuánto mejora la LSTM con las variables de estrategia respecto a la LSTM con la que empezó esta línea de trabajo, con las mismas métricas que el modelo final.

## Modelos (todos congelados antes de mirar 2025)

| Id | Modelo | Entradas | Umbral (fijado en 2024) | Validación 2024 |
|---|---|---|---:|---|
| `lstm_partida` | LSTM32, dropout 0,2, BCE `pos_weight=2`, semilla 42 (`runs/lstm32_pos_weight2_f2_cpu_v1_seed42.zip`) | v1_3 completo: 22 numéricas + piloto, equipo, GP | 0,12 | F2 0,462, AP 0,223 |
| `lstm_estrategia` | LSTM32 con la misma receta, media de las semillas 42, 43 y 44 (`modelos/lstm32_estrategia_v1/`) | Telemetría + 9 variables de estrategia por vuelta, sin contexto estático | 0,28 | F2 0,513, AP 0,343 |

`lstm_estrategia` se reentrenó con las configuraciones commiteadas (`configs/experimento_lstm32_estrategia_candidate_cpu_seed{42,43,44}_v1.json`) y reproduce exactamente las predicciones de las corridas anteriores (diferencia 0). `scripts/congelar_lstm_estrategia.py` fija el umbral de la media de semillas maximizando F2 en 2024, con la misma regla de todos los modelos. También se reportan las tres semillas por separado, cada una con su propio umbral de 2024.

## Métricas

Las mismas de la especificación 22: F2 por muestra (principal), AP, precisión, recall, cobertura por evento total y por tipo de parada, F2 con tolerancia N+3/N+4 y F2 por carrera. La población es la misma: 25.741 muestras elegibles de 2025.

## Ejecución

`scripts/evaluar_lstm_2025.py` reconstruye las entradas desde las ventanas causales crudas:

1. Preprocesador congelado de v1_3 (`transform`, sin ajuste).
2. Proyección de telemetría congelada. Se comprueba que el estado derivado es idéntico al publicado.
3. Variables de estrategia por vuelta, estandarizadas con las medias y escalas de train del paquete publicado.

Antes de 2025, el mismo código reconstruye 2024 y exige que las entradas sean idénticas a los tres paquetes publicados (v1_3, telemetría y estrategia) y que las probabilidades coincidan con las exportadas (tolerancia 10⁻⁵). El ensayo (`--solo-ensayo-2024`) se superó.

La LSTM de partida se exportó antes de que `lstm_model.py` admitiera capas adicionales; para ella la equivalencia funcional se garantiza con esa paridad exacta en 2024, no con el hash del fuente.

```bash
python scripts/evaluar_lstm_2025.py --confirmar-evaluacion-unica
```

Escribe en `docs/aclaraciones/evaluacion_lstm_2025/` y rechaza una carpeta existente. No se reentrena, no se cambian umbrales y no se excluyen carreras.
