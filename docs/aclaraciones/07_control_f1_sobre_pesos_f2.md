# Control F1: cambiar únicamente el umbral del checkpoint F2

**Estado:** control calculado sobre validación 2024. No hubo entrenamiento nuevo ni cambios en los pesos de la época 10. La referencia F2 quedó versionada en el commit local `cfcfb95`.

Este control separa el efecto de la regla de alerta del de elegir otro checkpoint. Usa las probabilidades exportadas por la primera RNN, la misma cuadrícula de umbrales 0,05–0,95 con paso 0,01 y el mismo desempate por precisión y mayor umbral.

| Criterio para elegir umbral con los mismos pesos | Umbral | F1 | F2 | Precisión | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| F2, referencia original | 0,12 | 0,27971 | 0,44463 | 17,28 % | 73,26 % | 8.116 | 619 |
| F1, control de decisión | 0,22 | **0,29640** | 0,42219 | 19,81 % | 58,88 % | 5.519 | 952 |

El control reduce 2.597 falsos positivos, a costa de 333 falsos negativos adicionales. Con la definición temporal original, cubre 552 de 824 eventos (66,99 %), frente a 652 de 824 (79,13 %) con la regla F2. No demuestra una mejora de la representación aprendida: solo cambió la decisión sobre los mismos scores. La precisión continúa siendo limitada.

El historial disponible de la primera corrida también alcanza su mayor F1 de validación a umbral fijo 0,5 en la época 10. Por ello, una corrida nueva que seleccione por F1 podría conservar el mismo checkpoint si reproduce la trayectoria. El ensayo F1 no garantiza otros pesos ni mejores resultados.

La [segunda configuración](../../configs/experimento_rnn_f1_v1.json) vuelve a entrenar con BCE, mismo modelo, datos, semilla y presupuesto. Selecciona checkpoint/parada temprana por F1 a 0,5 y después umbral por F1 con los pesos congelados. Compararemos su resultado con este control y con la referencia original; ver [especificación 12](../specs/12_experimento_rnn_f1.md).

## Evidencia y reproducción

La evidencia numérica y los hashes están en [control.json](evidencia_control_f1_v1/control.json). Las probabilidades fuente son `runs/rnn_v1_seed42/validation_predictions.csv.gz`; los pesos se mantienen con SHA-256 `bef673a6ad5349848b2a484c2563f86f65a1ced714af3810b7705c51481b6106`.

Para reproducir la selección del umbral desde la raíz del repositorio, después de incorporar el soporte de F1:

```python
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "src")
from f1_pitstop.experiment_metrics import select_metric_threshold

run = Path("runs/rnn_v1_seed42")
config = json.loads((run / "config.json").read_text())
predictions = pd.read_csv(run / "validation_predictions.csv.gz")
result = select_metric_threshold(
    predictions.target,
    predictions.probability,
    config["selection"]["threshold_grid"],
    metric="f1",
)
print(result)
```

Esta búsqueda usa validación y sirve para desarrollo; no es una medición independiente de generalización. No se evaluó ni transformó 2025. La corrida F1 nueva ya fue ejecutada y evaluada; ver [resultados y comparación](08_resultados_rnn_f1_y_comparacion.md).
