# Reporte de normalización histórica de Compound

- Fuente: `/home/simonsloan/Universidad/Semestre6/DeepLearning/F1-PitStop-Predictor/EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025.pkl`
- SHA-256: `9f8eb3f40d66869778c30d75f0a1ebc45196b8ad03c3a4258cc8f5e335dab65c`
- Coincide con evidencia v1: `True`
- Filas: 101,290; columnas: 61
- Conversiones: `None` → NA: 426; `nan` → NA: 354; total: 780
- Cambió únicamente `Compound`; no se reconstruyeron compuestos ni se tocaron etiquetas.
- Para este entorno (Python 3.14 + pandas 2.3.3), lea el pickle mediante `load_normalized_matrix()` para restaurar el dtype nullable `string`; el pickle guarda `Compound` como object con valores `pd.NA` por compatibilidad.
- Lea el CSV de auditoría con `load_normalization_audit()` para no interpretar el literal original `None` como nulo CSV.
- Esta es una matriz histórica normalizada, no una matriz de características aprobada para entrenar ni un reemplazo del replay causal `Compound_asof`.
