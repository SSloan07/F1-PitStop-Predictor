# Evaluación de los marcadores de Compound

**Estado:** normalización histórica de representación **acordada e implementada**; no aprueba una matriz de entrenamiento.
**Alcance:** representación de `Compound` en una futura tabla histórica derivada y relación con `Compound_asof`. No redefine etiquetas ni aprueba `Stint_asof`/`FreshTyre_asof`.

## Recomendación concreta

En una nueva tabla histórica derivada de la maestra, representar los literales exactos `"None"` y `"nan"` como **compuesto no determinado**, usando `pd.NA` en una columna de tipo nullable. La regla está implementada en `scripts/normalizar_compound.py`. El literal original y el motivo se conservan en una auditoría enlazada por `(Temporada, Ronda, DriverNumber, LapNumber)`. No se eliminan filas ni se asigna un compuesto físico por mayoría, por el neumático de otro stint o por observaciones futuras.

Esto normaliza dos representaciones de información no determinada; no afirma que ambas tengan idéntica causa ni que el vehículo no llevara neumáticos. La trazabilidad mantiene esa diferencia. El dato observado original permanece en la maestra.

La evidencia de inspección respalda esta regla de representación. Su ejecución escribe una salida nueva y rechaza sobrescribir artefactos. La salida es una matriz histórica normalizada, no una matriz de características autorizada para entrenamiento.

Para `X`, mantener el contrato ya acordado: `Compound_asof` se construye desde eventos elegibles hasta la decisión y usa `UNKNOWN` cuando no se conoce el compuesto. La columna histórica normalizada no reemplaza esa reconstrucción temporal. El literal y el motivo de auditoría no se convierten automáticamente en predictores.

## Evidencia observada

Se leyeron las 101.290 filas de la maestra, las candidatas de reconstrucción/limpieza anteriores, los cinco archivos de carrera afectados y sus cachés locales de `timing_app_data`. Los hashes antes/después coinciden para las 13 entradas. La maestra contiene 100.510 valores físicos (`HARD`, `MEDIUM`, `SOFT`, `INTERMEDIATE`, `WET`) y los siguientes 780 marcadores (0,770 % de filas):

| Carrera | Literal original | Filas | Diagnóstico |
|---|---|---:|---|
| Mónaco 2022 | `None` | 35 | Un stint de Stroll. La correspondencia diagnóstica fuente-matriz encuentra `INTERMEDIATE` y `HARD` en momentos distintos; la regla retrospectiva de compuesto único no puede aceptarlo automáticamente. |
| Austria 2022 | `None` | 299 | Doce stints. La candidata anterior reconstruye 298 filas no generadas; conserva una fila generada sin reconstruir. |
| Canadá 2023 | `None` | 35 | Un stint de Tsunoda; la candidata anterior reconstruye estas 35 filas. |
| Miami 2025 | `nan` | 354 | Quince pilotos sin `Stint` ni `TyreLife` en esas filas; todas terminan antes del primer mensaje de neumáticos conservado para cada piloto. |
| Bélgica 2025 | `None` | 57 | Cuatro stints; la candidata anterior reconstruye estas 57 filas. |

En total hay 18 stints identificados afectados y 15 bloques piloto-carrera sin stint identificado. No son 33 stints confirmados. No hay literales `UNKNOWN`, `TEST_UNKNOWN`, vacíos ni nulos reales de `Compound` en esta maestra. Esta propuesta no asigna tratamientos nuevos a categorías que no aparecen en ella.

### Caso que explica los `nan`: Miami 2025

Las 354 filas tienen `Stint` y `TyreLife` nulos. Para los quince pilotos, el primer registro de la caché está en `01:35:31.226` del reloj de sesión. Las 354 filas terminaron antes de esa marca; por tanto, la caché inspeccionada no aporta una actualización previa con la que recuperar causalmente su compuesto. Esto respalda tratarlas como desconocidas. No prueba que el feed original nunca hubiera publicado datos anteriores o que no puedan existir en otra fuente.

Ejemplo: Verstappen conserva `Compound="nan"` hasta la vuelta 24; el primer `MEDIUM` de la caché aparece a `01:35:31.226`, durante la vuelta 25. Copiar ese valor a las vueltas 1–24 introduciría información posterior a esas decisiones.

Además, las 354 filas conservan `FreshTyre=True` aunque carecen de actualización previa de neumáticos en la caché. Ese booleano consolidado no demuestra que el juego fuera nuevo y no permite rellenar `Compound` ni `TyreLife`. No se cambió este campo.

### Qué prueban las candidatas anteriores

La reconstrucción anterior cambia 390 `None` a compuestos físicos: 308 a `HARD` y 82 a `MEDIUM`. No cambia los valores de `Compound` que ya eran físicos. Quedan 36 `None` (35 Mónaco y una fila generada de Austria) y 354 `nan`. La limpieza v1 convierte esos 390 marcadores restantes en nulos.

Por eso hay dos cifras distintas: **780** valores no determinados si se normaliza la maestra y **390** si se parte de la candidata reconstruida. Las 390 recuperaciones son una reconstrucción histórica candidata: el conteo, la coincidencia entre archivos o la unicidad por stint no certifican por sí solos la disponibilidad causal ni autorizan esa versión como entrada de modelado.

## Diferencia entre tabla por vuelta y actualizaciones de la fuente

El parser oficial de [TimingAppData](https://github.com/theOehrly/Fast-F1/blob/main/fastf1/_api.py) describe `Compound` como texto o `None`; cuando un campo no aparece en una actualización, agrega `None`. Por ello, un nulo en un mensaje parcial no significa automáticamente que un compuesto previamente conocido deba borrarse. El código de [construcción de vueltas](https://github.com/theOehrly/Fast-F1/blob/main/fastf1/core.py) enlaza información de timing y neumáticos y define `Compound` como texto.

La presencia de literales en la matriz es compatible con una conversión de representaciones ausentes a texto, pero no se reconstruyó la versión exacta del paquete de adquisición para atribuirle con certeza ese origen. Las cachés declaran versión de parser 15, no versión del paquete. La recomendación no depende de demostrar esa conversión: los datos y fuentes inspeccionadas no sustentan interpretar `None`/`nan` como compuestos físicos.

Aplicar el reemplazo acordado **solo a los literales de la tabla histórica derivada**. No se aplica como borrado de estado en `timing_app_data`: un `None` en una actualización parcial puede significar campo omitido. En el replay se conserva el estado conocido dentro del mismo stint cuando el campo está omitido, se procesa la transición al stint activo y se evita heredar el compuesto del stint anterior. `Time` se interpreta como tiempo de evento bajo el supuesto temporal documentado.

## Controles de la propuesta

La ejecución materializada verifica la conservación de filas, claves, orden y las otras columnas; modifica únicamente `Compound`, mantiene los valores físicos y produce `pd.NA` para los marcadores exactos. Repetir la normalización no debe cambiar el resultado. Los hashes de las entradas se comprueban antes y después.

La clasificación usa evidencia de fuente y representación, sin etiquetas, métricas de modelos ni umbrales aprendidos. Se inspeccionaron también 2025 para calidad de fuente; esto no supone usar sus resultados predictivos para seleccionar un experimento.

## Evidencias y reproducción

- [Resumen y hashes de las entradas](evidencia_compound_v1/summary.json).
- [Las 780 filas afectadas y comparación con candidatas](evidencia_compound_v1/marker_rows.csv).
- [Registros físicos por bloque y timestamps de fuente](evidencia_compound_v1/stint_blocks.json).
- [Script de evaluación](evaluar_marcadores_compound.py), requiere pandas y rehúsa sobrescribir resultados.

Desde la raíz del repositorio, usando una ruta de salida nueva:

```bash
python3 docs/aclaraciones/evaluar_marcadores_compound.py \
  --output-dir /tmp/evaluacion_compound_nueva
```

La salida normalizada, su auditoría, manifiesto, conciliación y reporte se escriben en un directorio nuevo; ver el comando de reproducción en [la especificación de limpieza](../specs/04_limpieza_candidata.md). La recuperación de un compuesto físico se evalúa aparte y requiere fuente y disponibilidad compatibles con el uso previsto.
