# Comparación de la matriz candidata de compuestos

**Estado:** comparación reproducible completada; la matriz candidata aún no es la versión oficial de entrenamiento.
**Fuentes:** matriz maestra 2022–2025 y candidata producida por `EDA/dataset/reconstruir_compuestos_cache.py`.

## Integridad estructural

| Control | Resultado |
| --- | ---: |
| Filas en matriz original y candidata | 101.290 en ambas |
| Columnas y orden | 61; idénticos |
| Vueltas con `Stint`, `LapNumber` conocidos y no generadas por FastF1 | 100.801 |
| Claves duplicadas por temporada, carrera, piloto y vuelta | 0 |
| Filas cuyo `Compound` cambió | 390 |
| Otras columnas modificadas | 0 |
| `TyreLife` idéntico entre matrices | Sí |

Las únicas diferencias son 390 valores de `Compound` que estaban marcados como ausentes y ahora tienen una fuente asignada. No se reemplazó ningún compuesto previamente conocido, no se agregaron filas y no se modificaron clima, telemetría, etiquetas ni otras variables.

## Faltantes elegibles por temporada

Los faltantes se normalizaron considerando tanto nulos como marcadores textuales (`None`, `nan`, `UNKNOWN` y cadenas vacías). Por ello, el conteo bruto de nulos de pandas puede ser distinto.

| Temporada | Compound original | Compound candidato | TyreLife original | TyreLife candidato |
| ---: | ---: | ---: | ---: | ---: |
| 2022 | 333 | 35 | 518 | 518 |
| 2023 | 35 | 0 | 35 | 35 |
| 2024 | 0 | 0 | 0 | 0 |
| 2025 | 57 | 0 | 80 | 80 |
| **Total** | **425** | **35** | **633** | **633** |

Los 35 compuestos que siguen ausentes corresponden a un stint de Mónaco 2022 cuya caché ofrece evidencia ambigua. Los 633 `TyreLife` elegibles continúan ausentes: la fuente local utilizada no proporciona una edad directa que permita reconstruirlos con la misma validación.

En la matriz completa, el marcador de ausencia de `Compound` aparece como texto: 426 filas contienen `None` y 354 contienen `nan` en la original. La candidata reduce el total normalizado de 780 a 390; los 355 marcadores restantes están fuera del subconjunto elegible definido para esta reconstrucción.

## Dictamen

La candidata es estructuralmente íntegra y adecuada para continuar el **pipeline de limpieza como versión experimental candidata**. La evidencia apoya usarla para analizar el efecto de disponer de compuestos reconstruidos, pero aún no declara que deba reemplazar a la matriz maestra ni que esos valores sean predictores válidos al inicio de una vuelta. Esa disponibilidad temporal y la regla de limpieza final se deben decidir en el pipeline del experimento. Conservar ambas versiones y comparar resultados con la misma partición de carreras evita perder trazabilidad.

`TyreLife` debe mantenerse como ausente en esta candidata. La decisión sobre los 35 compuestos ambiguos de Mónaco también queda explícita: no se imputan.

## Reproducción

Ejecutar desde la raíz del repositorio:

```bash
python EDA/dataset/reconstruir_compuestos_cache.py
python EDA/dataset/comparar_matrices.py
```

El segundo script comprueba forma, columnas, claves, campos modificados, cobertura por temporada y faltantes elegibles restantes. La matriz candidata y su CSV de auditoría permanecen como artefactos locales, separados de los archivos de código y documentación.

## Relación con otros documentos

- [Validación de reconstrucción de neumáticos](04_validacion_reconstruccion_neumaticos.md): método y contraste con stints conocidos.
- [Reglas candidatas de limpieza](../specs/04_limpieza_candidata.md): decisiones generales pendientes para la matriz de entrenamiento.
