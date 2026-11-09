# LSTM32 de una frente a dos capas: resultados

**Estado:** pareja completada; una semilla (42); validación 2024. La prueba 2025 no se utilizó.

La comparación cambió únicamente el número de capas recurrentes. Ambos modelos se entrenaron desde cero con la misma partición, preprocesador, BCE normal, dropout 0,2 en la cabeza, selección por F2 y entorno CPU. Los hashes del paquete y de los 42 artefactos del ZIP pasaron; ambos hijos terminaron y el runner autorizó la comparación.

| Métrica de validación | LSTM32 · 1 capa | LSTM32 · 2 capas | Diferencia |
|---|---:|---:|---:|
| Época seleccionada | 20 | 17 | — |
| Umbral seleccionado por F2 | 0,08 | 0,07 | — |
| F2 | 0,45284 | 0,45810 | +0,00526 |
| F1 | 0,28653 | 0,30188 | +0,01536 |
| Precisión | 17,77 % | 19,25 % | +1,47 pp |
| Recall por muestra | 73,87 % | 69,94 % | −3,93 pp |
| AP | 0,20134 | 0,21652 | +0,01518 |
| Falsos positivos | 7.911 | 6.792 | −1.119 |
| Falsos negativos | 605 | 696 | +91 |
| Eventos cubiertos | 663/824 (80,46 %) | 642/824 (77,91 %) | −21 eventos |
| Parámetros | 12.377 | 20.825 | +8.448 (+68,3 %) |

La candidata de dos capas eleva ligeramente F2 y AP y reduce falsas alertas por muestra, a cambio de perder recall y dejar sin cubrir 21 eventos más. El F2 de ambos sigue cerca de 0,46 y la precisión por debajo de 20 %; por tanto, el resultado aún no alcanza la meta provisional de F2 ≥ 0,60, precisión ≥ 30 % y recall ≥ 80 %. Es una mejora exploratoria, no evidencia de utilidad operativa.

El evaluador local reprodujo las probabilidades exportadas: diferencia máxima 3,58×10⁻⁷ en la referencia y 4,77×10⁻⁷ en la candidata; cero decisiones distintas en ambos casos. La cobertura por evento considera las 824 oportunidades evaluables, con la regla vigente del horizonte N–N+2 y entrada posterior al instante causal. Son métricas de validación usadas durante selección, no prueba independiente.

La conclusión está limitada a una semilla. No declaramos ganadora a la LSTM de dos capas; repetir la pareja con semillas adicionales es necesario para saber si la pequeña subida de F2 se sostiene. La prueba 2025 permanece reservada.

## Artefactos y procedencia

- Corrida Colab: `runs/lstm32_lstm2layer_f2_cpu_v1_seed42.zip`.
- SHA-256: `99e372b89d994146bceebd00c07c78b5e77b748b6dfba39305d9980e44b7d8db`.
- Evaluación local y evidencia resumida: [`evidencia_lstm_depth_v1.json`](evidencia_lstm_depth_v1.json).
- Manifiesto de pareja: incluido en el ZIP; `comparison_authorized=true`, `test_used=false`.
