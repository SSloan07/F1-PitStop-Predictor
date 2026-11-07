# Limpieza inicial v1

La salida histórica se genera con `python EDA/dataset/limpiar_matriz.py` a partir de la candidata de compuestos. Conserva 101.290 filas y 61 columnas; transforma 390 marcadores textuales `None`/`nan`/vacíos de `Compound` en valores ausentes de pandas. No imputa `TyreLife` ni elimina filas. Las claves incompletas o duplicadas detienen la ejecución.

La ejecución guarda la matriz `vueltas_2022_2025_limpieza_v1.pkl`, un informe JSON con huellas SHA-256 y un CSV de cobertura por carrera y piloto. Se observan 390 compuestos y 989 edades de neumático ausentes en la matriz completa, y cero saltos mayores a una vuelta entre registros consecutivos de un piloto. Esto no prueba que no falten vueltas al inicio o final.

Una comparación independiente verificó que las demás columnas y los compuestos conocidos permanecen idénticos. El script no sobrescribe una versión existente. Los artefactos son locales y se generan sin llamadas a la API.

Esta es una limpieza inicial histórica, no una matriz lista para entrenamiento: siguen pendientes el etiquetado de cambios de neumáticos, la alineación temporal de clima y telemetría y la comprobación de cuándo se conocieron los compuestos reconstruidos. Las transformaciones que aprenden estadísticas deben ajustarse con entrenamiento únicamente.
