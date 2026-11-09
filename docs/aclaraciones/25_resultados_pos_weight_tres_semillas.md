# BCE con `pos_weight=2`: resultado de tres semillas

**Estado:** comparación exploratoria completada en validación; la mejora no es consistente. La prueba final 2025 no se utilizó.

Se compararon tres pares desde cero (semillas 42, 43 y 44) en CPU. Cada pareja mantuvo LSTM32, dropout de fusión 0,2, datos, selección de checkpoint y umbral por F2, y cambió solo BCE normal frente a `pos_weight=2`. Métricas al umbral seleccionado en validación 2024:

| Semilla | BCE normal F2 | pos_weight=2 F2 | ΔF2 | BCE normal AP | pos_weight=2 AP | ΔAP | Eventos cubiertos (normal → ponderada) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.45196 | 0.46223 | +0.01027 | 0.20146 | 0.22342 | +0.02196 | 664/824 → 666/824 |
| 43 | 0.44556 | 0.45525 | +0.00969 | 0.22157 | 0.24022 | +0.01865 | 642/824 → 652/824 |
| 44 | 0.46214 | 0.45596 | -0.00618 | 0.21439 | 0.19739 | -0.01700 | 668/824 → 662/824 |

Promedio F2: 0.45322 → 0.45781; Δ media +0.00459; signo positivo en 2/3 semillas.

Promedio AP: 0.21247 → 0.22034; Δ media +0.00787; signo positivo en 2/3 semillas.

La candidata mejora F2 y AP en dos de las tres semillas, pero empeora ambas en la tercera. Los cambios de cobertura por evento también varían (+2, +10 y −6). Con tres semillas no afirmamos superioridad general; mantenemos BCE normal como referencia. La cobertura por evento se obtuvo de la evaluación local con paridad y no está incluida en los ZIP de entrenamiento. Todos los pares marcan `comparison_authorized=true` y `test_used=false`.

Archivos fuente: `runs/lstm32_pos_weight2_f2_cpu_v1_seed42.zip` y `runs/lstm32_pos_weight2_f2_cpu_replicates_43_44.zip`; la vista del árbol registra sus SHA-256.
