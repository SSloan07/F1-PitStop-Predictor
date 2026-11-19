# LSTM en 2025: cuánto mejoró la LSTM con estrategia (comparación a posteriori)

**Estado:** evaluación única ejecutada el 2026-11-19 según la [especificación 23](../specs/23_comparacion_lstm_2025.md), declarada y commiteada antes de mirar 2025. Es una **comparación a posteriori**: el resultado principal del proyecto sigue siendo el del LightGBM pre-registrado ([informe 35](35_resultado_final_test_2025.md)). No se reentrenó ni se cambió ningún umbral con 2025.

Artefactos: `docs/aclaraciones/evaluacion_lstm_2025/`. Antes de 2025, el script reconstruyó 2024 y reprodujo exactamente los tres paquetes de entrada publicados y las probabilidades exportadas.

## Resultados en 2025

| Prueba 2025 | LSTM de partida | **LSTM + estrategia** (media de 3 semillas) | LightGBM final (referencia) |
|---|---:|---:|---:|
| Umbral (fijado en 2024) | 0,12 | 0,28 | 0,14 |
| **F2 (principal)** | 0,411 | **0,484** | 0,474 |
| AP | 0,201 | 0,299 | **0,332** |
| F1 | 0,251 | 0,329 | **0,341** |
| Precisión | 15,2 % | 21,4 % | **23,3 %** |
| Recall | **71,6 %** | 70,7 % | 64,0 % |
| Falsas alertas | 9.034 | 5.870 | **4.767** |
| Paradas no detectadas | **642** | 662 | 814 |
| Paradas cubiertas (de 820) | **628** | 610 | 613 |
| — en verde, stint normal | **85,3 %** | 84,5 % | 82,9 % |
| — SC/VSC, stint normal | 60,7 % | 60,2 % | **68,9 %** |
| — stint corto | **56,8 %** | 40,7 % | 34,6 % |
| F2 con tolerancia N+3/N+4 | 0,432 | **0,513** | 0,500 |
| F2 por carrera (mín. / mediana / máx.) | 0,01 / 0,41 / 0,61 | 0,14 / 0,50 / 0,72 | 0,16 / 0,48 / 0,74 |

Semillas de la LSTM con estrategia por separado (cada una con su umbral de 2024): F2 0,449, 0,485 y 0,462; AP 0,256, 0,323 y 0,278. El promedio de semillas (F2 0,484) supera a dos de las tres y empata con la mejor.

## De 2024 a 2025

| F2 | Validación 2024 | Prueba 2025 | Caída |
|---|---:|---:|---:|
| LSTM de partida | 0,462 | 0,411 | −0,051 |
| LSTM + estrategia | 0,513 | 0,484 | −0,029 |
| LightGBM final | 0,540 | 0,474 | −0,066 |

## Lectura

1. **Las variables de estrategia mejoran claramente a la LSTM en la temporada no vista:** F2 0,411 → 0,484 (+0,073, +18 %), AP 0,201 → 0,299 (+49 %) y 3.164 falsas alertas menos (−35 %), con casi el mismo recall (71,6 % → 70,7 %). La mejora vista en 2024 se mantiene en 2025.
2. **En 2025, la LSTM con estrategia empata o supera ligeramente al LightGBM en F2** (0,484 frente a 0,474) y en F2 con tolerancia, aunque ordena peor (AP 0,299 frente a 0,332). La diferencia de F2 (+0,010) es menor que la variación entre semillas de la LSTM (0,449–0,485), así que **no se puede declarar a ninguno mejor**. Lo que sí cambia es el punto de operación: la LSTM da más recall (70,7 % frente a 64,0 %) con más falsas alertas (5.870 frente a 4.767).
3. **La LSTM con estrategia perdió menos al pasar a 2025** (−0,029 frente a −0,066). Una explicación plausible: el LightGBM acumuló más decisiones tomadas mirando 2024 (rondas de variables 2 y 3, ajuste de hiperparámetros), así que su número de validación era más optimista.
4. La LSTM de partida cubre más paradas de stint corto (56,8 %) porque su umbral bajo (0,12) dispara muchas más alertas: 9.034 falsas alertas, casi el doble que la LSTM con estrategia.

## Conclusión

- Para un proyecto de Redes Neuronales, **la LSTM con variables de estrategia es un modelo final defendible**: en 2025 obtiene F2 0,484 (frente a 0,411 de la LSTM de partida), al nivel del LightGBM.
- Esta comparación es a posteriori. El resultado pre-registrado sigue siendo el del LightGBM (F2 0,474). La LSTM debe presentarse como comparación declarada después, con su especificación commiteada antes de ejecutarla.
- Ninguno alcanza la meta provisional (F2 ≥ 0,60).
