# Etiquetado candidato: entrada a boxes

La etiqueta predice si se registra una visita a boxes dentro de las tres vueltas incluyendo la actual. No exige comprobar un cambio de neumáticos. `PitInTime` ancla el evento; `PitOutTime` lo empareja. Una entrada sin salida queda desconocida aunque una fuente externa la reporte; esas fuentes auditan discrepancias y no sustituyen la pareja.

- Eventos detectados: 3469 (3359 clasificados; 110 sin resolver).
- Casos revisados: 33; reclamos de entrada respaldados por FIA: 28; por otras fuentes: 1; cruce de fuentes: 0.
- Casos pendientes con diagnóstico local: 110 (`review_queue.csv`).
- Etiquetas: 9393 positivas, 88163 negativas, 3734 desconocidas.
- Temporadas cubiertas: 2022, 2023, 2024, 2025.
- Invariancia 2022–2023 frente a pit_entry_v2: 47997 claves idénticas.
- No se entrenó ningún modelo ni se ajustaron preprocesadores.
