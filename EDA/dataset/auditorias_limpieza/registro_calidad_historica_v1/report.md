# Registro de calidad de la matriz histórica

- Entrada publicada: `/home/simonsloan/Universidad/Semestre6/DeepLearning/F1-PitStop-Predictor/EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_compound_normalizado_v1/vueltas_2022_2025_compound_normalizado.pkl`
- SHA-256 entrada: `02c15d7b2d766e1d01604cf143bb288a2e584370f0f422766683866a10ca00e0`
- Filas/columnas: 101,290/61; claves nulas/duplicadas: 0/0.
- Filas de registro: 16,644; razones distintas: 17.
- Compound: 780 ausencias enlazadas a auditoría (426 literal `None`, 354 `nan`).
- TyreLife/Stint: se conservan 989/354 faltantes; no se imputan.
- SpeedFL: los faltantes quedan dentro del contexto PitInTime/filas generadas; 2 entradas PitIn conservan SpeedFL y quedan señaladas. Position falta exactamente en filas generadas.
- Acelerador filtrado: validación de conteos, fracción, NA y rango 0–100 aprobada.
- Valores brutos, ceros, casos generados y excepciones temporales se registran sin modificarlos.
- Discrepancias LapTime/sectores: 14; coincidencia exacta con caché extendida: 14.
- Solapamientos Time/LapStartTime: 22; Monaco 2024 queda explicado por reinicio tras bandera roja; Saudi 2023 sigue sin resolver.
- No se escribió otra matriz, no se alteraron etiquetas y no se cambió ningún valor.
- Los casos sin fuente independiente suficiente quedan preservados y señalados; no se aplican reparaciones por intuición.

## Política temporal

`Time` de FastF1 respalda replay histórico por tiempo de evento, no prueba latencia de recepción en vivo. Clima, estados agregados y resúmenes completos de vuelta siguen siendo históricos; su disponibilidad as-of se construirá después.
