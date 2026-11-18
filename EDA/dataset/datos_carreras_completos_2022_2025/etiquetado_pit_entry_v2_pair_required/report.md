# Etiquetado candidato: entrada a boxes

La etiqueta predice si se registra una visita a boxes dentro de las tres vueltas incluyendo la actual. No exige comprobar un cambio de neumáticos. `PitInTime` ancla el evento; `PitOutTime` lo empareja. Una entrada sin salida queda desconocida aunque una fuente externa la reporte; esas fuentes auditan discrepancias y no sustituyen la pareja.

- Eventos detectados: 1779 (1715 clasificados; 64 sin resolver).
- Casos revisados: 33; reclamos de entrada respaldados por FIA: 28; por otras fuentes: 1; cruce de fuentes: 0.
- Casos pendientes con diagnóstico local: 64 (`review_queue.csv`).
- Etiquetas: 4817 positivas, 41361 negativas, 1819 desconocidas.
- Las etiquetas son candidatas; no se usaron 2024–2025 y no se entrenó ningún modelo.
