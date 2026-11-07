# Ejecución inicial del etiquetado candidato

Se ejecutó el script de etiquetado sobre la matriz de limpieza v1, filtrando las temporadas 2022–2023 antes de construir eventos o seleccionar casos.

La ejecución encontró 1.779 entradas, 49 salidas sin pareja identificable y seleccionó 32 eventos para revisión determinista. Produjo 47.997 muestras: 41.361 negativas según la cobertura disponible, 6.636 desconocidas y cero positivas. Las negativas se limitan a horizontes con las tres vueltas observadas, no generadas y tiempos coherentes; no representan confirmación física independiente.

La regla automática R1 permanece deshabilitada (estado requires_review): no se han añadido decisiones externas en el archivo review_decisions.json. Por ello, esta salida es un inventario de trabajo y una base para completar la revisión, no un conjunto de entrenamiento. Las etiquetas positivas solo aparecerán al registrar eventos verificados o tras habilitar R1 cumpliendo los controles.

Artefactos locales:

- events.csv: evidencia observada por cada entrada detectada.
- orphan_exits.csv: salidas que no se pudieron emparejar.
- review_cases.json: selección de casos con semilla 42.
- labels.pkl y coverage.csv: etiquetas triestado y cobertura por carrera.
- manifest.json y report.md: versión, huella de entrada y resumen.

Verificación ejecutada: compilación del script, claves piloto-vuelta únicas, tipo nullable de la etiqueta, temporadas limitadas y coherencia del número de casos seleccionados. La siguiente operación es completar la revisión externa del muestreo y volver a ejecutar el pipeline; no ajustar manualmente etiquetas a partir de una regla no habilitada.
