# Contrato de preprocesamiento de entradas

**Versión:** 1.0 · **Estado:** implementado para `muestras_causales_v1_3`, etiquetas v3 y particiones v1_1.
**Alcance:** ajuste de transformaciones con entrenamiento supervisado elegible y aplicación congelada a train/validación. No define arquitecturas, pérdida, embeddings, umbrales ni evaluación predictiva.

## 1. Entradas y límites

- Características, contexto, orden y máscaras: `muestras_causales_v1_3`.
- Etiquetas v3 y particiones: `muestras_causales_v1_3_etiquetas_v3_particiones_v1_1`.
- Etiquetas vigentes: `etiquetado_pit_entry_v3_2022_2025`.
- La columna `labels_aligned.csv.gz` dentro de v1_3 es histórica y no se usa.
- `sample_id`, claves, W=8 y orden de campos deben coincidir en índices, contexto, etiquetas y arrays. Si un hash, esquema, dimensión o versión no coincide, la ejecución falla antes de publicar.
- Se transforman todas las muestras de train y validación para conservar alineación y elegibilidad; las estadísticas se ajustan únicamente con filas `split=train AND supervised_eligible=true`.
- Para la prueba final 2025 solo se verifican metadatos de pertenencia, claves y separación por carrera. No se leen sus valores objetivo para el ajuste, no se calculan estadísticas/coberturas de sus características y no se genera un array transformado de prueba.

## 2. Variables numéricas dinámicas

La lista y el orden se toman del `feature_registry.json` de v1_3; actualmente son 22 campos. Para cada característica, la media y desviación poblacional (`ddof=0`) se estiman en `float64` con las celdas cuyo código de motivo sea `present`, y cuya posición tenga `slot_exists_mask=true` y `padding_mask=false`. Las máscaras originales deben concordar con los códigos; en caso contrario, se rechaza la entrada.

El cero almacenado en una posición enmascarada no participa del ajuste. Un cero físico con máscara `present` sí participa. Si la varianza es exactamente cero, la escala queda en 1 y se registra el campo como constante. Una variable sin observaciones válidas en train detiene la publicación con el nombre de la variable; no se inventa una media. No hay interpolación, imputación, relleno temporal ni reconstrucción de medidas. Tras estandarizar, los valores no observados se guardan como cero técnico y su motivo/máscaras originales se conservan por separado.

Las ventanas se solapan. Cada aparición de una celda válida en una ventana de entrenamiento elegible aporta una observación al ajuste. Por ello, si una medición de vuelta aparece en varias ventanas, se pondera tantas veces como apariciones tenga; no se deduplica por vuelta de origen.

## 3. Variables categóricas

Se mantienen dos ramas de IDs, sin embeddings ni arquitectura definidos:

- **Contexto:** `Driver`, `Team`, `GranPremio`, una fila por muestra.
- **Secuencia:** `Compound_asof`, `track_phase_asof`, por posición de ventana.

Los vocabularios se aprenden solo de muestras supervisadas elegibles de train. Los valores nuevos en validación se codifican como fuera de vocabulario y no amplían el estado ajustado. Los valores se ordenan por punto de código Unicode, con IDs consecutivos deterministas.

| ID | Representación |
|---:|---|
| 0 | Padding de posición temporal. No se usa para ausencia de campo. |
| 1 | Ausencia: valor nulo/blank o enmascarado; también una posición existente sin fila fuente. |
| 2 | Literal fuente `UNKNOWN`, cuando la posición existe y no está bloqueada por padding o indisponibilidad temporal. |
| 3 | Categoría observada y disponible que no aparece en el vocabulario ajustado. |
| 4 en adelante | Categorías conocidas, en orden Unicode determinista. |

Prioridad dinámica: padding → ausencia estructural de slot → `UNKNOWN` explícito con código de fuente desconocida (o `present`) → ausencia indicada por máscara/campo → vocabulario conocido → fuera de vocabulario. Las máscaras permanecen independientes y permiten distinguir las causas de ausencia que comparten el ID 1. Para el contexto estático no existe padding ni máscara temporal: nulo/blank → ausencia, literal `UNKNOWN` → desconocido fuente, vocabulario conocido → ID aprendido, otro valor → fuera de vocabulario.

Los textos exactos vacíos, `nan`, `none` y `<na>` (comparación case-insensitive) se consideran representaciones ausentes de contexto. Los demás literales se preservan; `UNKNOWN` es reservado y no se aprende como categoría ordinaria.

## 4. Máscaras y alineación

Se preservan sin colapsar: `padding_mask`, `slot_exists_mask`, `slot_row_missing_mask`, `temporal_unavailable_mask`, `feature_missing_mask` y `mask_reason_codes`; también se conservan `window_lap_numbers` y `lineage_ids` para auditoría. No se concatenan automáticamente al vector predictor. En el NPZ se almacenan como arrays separados de `X_numeric`, IDs dinámicos e IDs estáticos.

Las etiquetas se exportan en tablas laterales, nunca dentro de `X`. Cada NPZ conserva `sample_id` y el orden original del split; los archivos de metadatos y etiquetas se validan contra el mismo orden. La tabla train conserva unknowns e inelegibles aunque no participen del ajuste; el consumidor puede seleccionar `supervised_eligible` para la pérdida.

## 5. API y publicación

`InputPreprocessor.fit(...)` es de una sola ejecución y requiere un selector explícito de población autorizada. `transform(...)` valida dimensiones, W, nombres/orden, tipos y contrato de máscaras, y no modifica el estado ajustado. `save/load` serializa JSON sin pickle, con conteos, medias, escalas, cardinalidades y vocabularios.

La CLI fija hashes de los manifiestos de entradas, transforma train y validación con el mismo estado, y rechaza la carpeta de salida si ya existe. Salida vigente: `EDA/dataset/muestras_preprocesadas_v1_3/`. Las publicaciones `_v1`, `_v1_1` y `_v1_2` se conservan sin modificar.

Reproducción:

```bash
python scripts/preparar_preprocesador.py \
  --config configs/preprocesador_entradas_v1.json
```

El paquete incluye `example_load.py`, que carga el estado congelado y vuelve a transformar validación desde los arrays originales, comprobando igualdad con los artefactos publicados. Se ejecuta desde la raíz del repositorio con `python EDA/dataset/muestras_preprocesadas_v1_3/example_load.py`.

No se generó un array transformado, estadísticas de características ni cobertura de valores para prueba. Solo se verificaron sus metadatos de pertenencia, claves y separación. El método reutilizable `transform` permite en una fase futura aplicar el estado guardado a test o producción, tras la aprobación del protocolo correspondiente.
