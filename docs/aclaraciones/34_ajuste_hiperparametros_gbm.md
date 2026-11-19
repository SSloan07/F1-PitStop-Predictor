# Ajuste de hiperparámetros de LightGBM

**Estado:** completado. La búsqueda y la selección usan solo train (2022–2023); la validación 2024 se usó una vez para confirmar. La prueba 2025 no se cargó.

## Pregunta

Hasta ahora LightGBM usaba parámetros razonables pero fijos. Con el mejor conjunto de entradas (telemetría + estrategia de rondas 1–3, 66 variables; ver [ronda 3](33_variables_numero_paradas.md)), ¿mejora con un ajuste de hiperparámetros?

## Protocolo

- **Candidatas:** la configuración actual más 30 configuraciones aleatorias (semilla 2026) de esta cuadrícula declarada:

  | Hiperparámetro | Valores |
  |---|---|
  | `num_leaves` | 7, 15, 31, 63, 127 |
  | `min_child_samples` | 20, 50, 100, 200, 400 |
  | `learning_rate` | 0,01; 0,02; 0,03; 0,05 |
  | `feature_fraction` | 0,5; 0,7; 0,8; 1,0 |
  | `bagging_fraction` | 0,6; 0,8; 1,0 |
  | `lambda_l2` | 0, 1, 5, 20 |
  | `min_gain_to_split` | 0; 0,1; 0,5 |
  | `max_bin` | 63, 255 |

- **Puntuación:** AP medio fuera de partición en las mismas 5 particiones por carrera de train, con parada temprana en el 20 % de las carreras de cada partición de entrenamiento (hasta 5.000 árboles, paciencia 150).
- **Regla fijada antes de ver resultados:** se adopta la mejor si supera a la actual por más de 0,002 de AP medio y en al menos 4 de 5 particiones.
- **Confirmación:** una sola vez en 2024. Cada configuración es la media de tres modelos (semillas 42, 43 y 44); el número de árboles se elige con validación cruzada en train y el umbral por F2 en la cuadrícula de siempre.

```bash
python scripts/ajuste_gbm.py --output docs/aclaraciones/evidencia_ajuste_gbm_v1
```

## Resultados en train

| Configuración | AP medio por partición | AP fuera de partición |
|---|---:|---:|
| Actual (31 hojas, mín. 50, lr 0,03) | 0,3605 | 0,3534 |
| **Mejor: `random_06`** | **0,3724** | **0,3682** |
| Segunda: `random_03` | 0,3721 | 0,3666 |

`random_06` mejora en **las 5 particiones** (+0,004, +0,011, +0,017, +0,023, +0,004; media +0,012). **Se adopta.**

Configuración adoptada: `num_leaves=127`, `min_child_samples=100`, `learning_rate=0,01`, `feature_fraction=0,7`, `bagging_fraction=0,6`, `lambda_l2=5`, `min_gain_to_split=0,5`, `max_bin=63` (unos 300–330 árboles). Las mejores configuraciones coinciden en el patrón: árboles más grandes, pero con hojas mínimas grandes, submuestreo de filas y variables, más regularización y una tasa de aprendizaje baja.

## Confirmación en validación 2024

| Validación 2024 (media de 3 semillas) | AP | F2 | F1 | Precisión | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Configuración actual | 0,3859 | 0,5352 | 0,4010 | 28,3 % | 68,9 % | 4.046 | 720 |
| **Configuración ajustada** | **0,3867** | **0,5399** | 0,3994 | 27,9 % | **70,5 %** | 4.229 | **682** |

- En 2024 la mejora es pequeña: AP +0,001 y F2 +0,005. Es bastante menor que en train (+0,012 de AP), algo esperable cuando se elige la mejor de 31 configuraciones.
- La configuración ajustada es más estable entre semillas en 2024: AP individual 0,381–0,388, frente a 0,378–0,387 de la actual.
- Promediar tres semillas y la parada temprana más paciente ya subieron la configuración actual de AP 0,378 (una semilla, [ronda 3](33_variables_numero_paradas.md)) a 0,386.

## Conclusiones

1. El ajuste de hiperparámetros sí ayuda, pero poco: +0,005 de F2 en 2024. El rendimiento lo limitan sobre todo las variables y el tipo de error descrito en el [análisis de errores](32_analisis_errores.md), no la capacidad del modelo.
2. **Mejor modelo de desarrollo hasta ahora:** LightGBM ajustado, media de 3 semillas, 66 variables, umbral 0,14: **AP 0,387, F2 0,540, precisión 27,9 %, recall 70,5 %** en 2024.
3. La validación 2024 se ha usado en muchas decisiones. Estas cifras son de desarrollo y probablemente algo optimistas.

## Siguiente paso recomendado

Congelar el modelo final (variables de rondas 1–3, configuración `random_06`, media de semillas 42–44, umbral 0,14 elegido en 2024) y las métricas que se reportarán antes de mirar 2025. Después, extender el pipeline causal a 2025 y evaluar una sola vez.
