# Ensamble LightGBM + LSTM en validación 2024

**Estado:** completado en validación 2024 con las corridas LSTM locales preliminares (CPU, semillas 42–44). Combinación fijada de antemano, sin pesos ajustados. La prueba 2025 no se cargó.

## Pregunta

LightGBM con estrategia de ronda 1 + ronda 2 es el mejor modelo hasta ahora (AP 0,369, F2 0,534), y la mejor LSTM es la de estrategia de ronda 1 (AP medio 0,313, F2 0,503; [especificación 20](../specs/20_experimento_lstm_estrategia.md)). ¿Combinar sus probabilidades mejora sobre cada uno?

## Protocolo

Regla fijada antes de ver resultados:

- **GBM:** promedio de probabilidades de tres LightGBM (semillas 42, 43 y 44) con las 61 variables de base + todos los grupos de la ronda 2. El número de árboles se elige con `GroupKFold` por carrera en train.
- **LSTM:** promedio de las probabilidades de validación exportadas por las tres LSTM32 de estrategia ronda 1.
- **Ensamble (principal):** 0,5 × GBM + 0,5 × LSTM.

Cada candidato elige su umbral por F2 en la cuadrícula congelada de validación, igual que en todos los experimentos, por lo que todas son métricas de desarrollo. La cobertura por evento usa la lógica del evaluador compartido.

```bash
python scripts/ensamble_gbm_lstm.py --output docs/aclaraciones/evidencia_ensamble_v1/resultados
```

Las predicciones LSTM de entrada se guardan en `evidencia_ensamble_v1/entradas_lstm/` con sus hashes en el manifiesto.

## Resultados

| Validación 2024 | AP | F2 | Precisión | Recall | Eventos cubiertos (de 824) |
|---|---:|---:|---:|---:|---:|
| GBM semilla 42 | 0,3693 | 0,5344 | 28,4 % | 68,6 % | 641 |
| GBM semilla 43 | 0,3642 | 0,5297 | 26,9 % | 69,8 % | 648 |
| GBM semilla 44 | 0,3671 | 0,5369 | 26,6 % | 72,1 % | 666 |
| LSTM semilla 42 | 0,2860 | 0,4812 | 22,6 % | 67,0 % | 618 |
| LSTM semilla 43 | 0,3240 | 0,5184 | 24,5 % | 71,9 % | 654 |
| LSTM semilla 44 | 0,3295 | 0,5089 | 25,2 % | 68,3 % | 638 |
| **GBM media de 3** | 0,3701 | **0,5353** | **28,5 %** | 68,6 % | 640 |
| **LSTM media de 3** | 0,3426 | 0,5130 | 24,9 % | 69,9 % | 643 |
| **Ensamble 50/50** | **0,3783** | 0,5294 | 25,0 % | **73,5 %** | **672** |

Correlación entre las probabilidades de GBM y LSTM: 0,83.

## Lectura

1. **Promediar semillas ayuda mucho a la LSTM:** AP 0,313 (media de modelos individuales) → 0,343 (modelo promedio), F2 0,503 → 0,513. La LSTM individual es inestable entre semillas y el promedio reduce esa varianza. En LightGBM, promediar semillas apenas cambia (0,367 → 0,370).
2. **El ensamble tiene el mejor AP** (0,378, +0,008 sobre GBM) y **la mayor cobertura por evento** (672 de 824, +32 sobre GBM), con recall por muestra de 73,5 %.
3. **En F2 por muestra no gana:** 0,529 frente a 0,535 del GBM. Al umbral elegido, cubre más eventos con más falsas alertas (precisión 25,0 % frente a 28,5 %).
4. La correlación de 0,83 indica que los modelos se equivocan de forma parecida: el margen del ensamble es limitado.

## Conclusión

No hay un ganador claro en la métrica principal. **GBM (media de semillas)** tiene el mejor F2 por muestra y la mejor precisión. El **ensamble 50/50** ordena mejor las muestras (AP) y anticipa más paradas reales. La diferencia de F2 (0,006) es menor que la variación entre semillas de GBM (≈0,007).

La elección depende del uso:

- Para **menos falsas alertas** (por ejemplo, en el replay): GBM.
- Para **no perder paradas**: el ensamble.

En cualquier caso, la LSTM debería usarse como promedio de semillas, no como modelo individual.

## Límites

- Las LSTM son corridas locales; faltan las oficiales de Colab.
- La validación 2024 ya se ha usado varias veces para decisiones (variables de ronda 1, confirmación de ronda 2, este ensamble). La comparación final debería hacerse una sola vez en el test 2025, con el modelo y el umbral congelados antes.
- Seguimos debajo de la meta provisional (F2 ≥ 0,60, precisión ≥ 30 %, recall ≥ 80 %).
