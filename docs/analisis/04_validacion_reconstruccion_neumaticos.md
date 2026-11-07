# Validación de reconstrucción de datos de neumáticos

**Estado:** evaluación retrospectiva y matriz candidata generada; la matriz original permanece intacta.
**Matriz analizada:** `EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025.pkl` (101.290 filas).

## Pregunta

¿Se pueden reconstruir `Compound` y `TyreLife` a partir de valores adyacentes sin crear filas duplicadas ni cruzar cambios de stint?

## Método

Se analizaron filas con `LapNumber` y `Stint` conocidos, excluyendo `FastF1Generated=True`. La unidad de agrupación fue temporada, carrera, piloto y stint. Para la validación retrospectiva se ocultó cada valor conocido por turno y se intentó recuperarlo con los valores observados más cercanos dentro de un máximo de dos vueltas. Para `Compound`, dos vecinos debían coincidir; si solo había un vecino disponible, se podía usar ese valor. Para `TyreLife`, se estimó la edad según la distancia en vueltas entre los vecinos conocidos. No se cruzaron stints.

Esta prueba mide huecos aislados en stints con observaciones conocidas. No simula directamente un stint entero sin datos, por lo que sus métricas de recuperación no se deben extrapolar a bloques faltantes.

## Resultados

| Variable | Valores conocidos evaluados | Recuperables al ocultarlos | Cobertura | Resultado en los recuperables |
| --- | ---: | ---: | ---: | --- |
| `Compound` | 100.376 | 100.014 | 99,64 % | 100 % de coincidencia |
| `TyreLife` | 100.168 | 99.807 | 99,64 % | MAE de 0 vueltas; 100 % exacto |

La recuperación retrospectiva de huecos puntuales funcionó muy bien cuando había referencias cercanas en el mismo stint. Sin embargo, los faltantes observados tienen una estructura distinta:

| Variable | Filas faltantes elegibles | Stints afectados | Stints completamente faltantes |
| --- | ---: | ---: | ---: |
| `Compound` | 425 | 18 | 18 |
| `TyreLife` | 633 | 41 | 41 |

En los stints afectados no hay valores conocidos de esa variable que sirvan como ancla dentro del mismo stint. La reconstrucción por vecinos no puede completar esos bloques. Tomar valores de stints anteriores o posteriores podría atravesar una parada y asignar el neumático equivocado.

## Contraste con la caché de timing app

Las cachés locales de FastF1 contienen `timing_app_data`, que a veces registra el compuesto al iniciar un stint aunque las vueltas de ese stint no lo tengan. El `Stint` de esa fuente se alinea con el de la matriz usando el índice de timing app menos uno. La correspondencia se valida por carrera contra los stints con compuesto observado. Solo se acepta una reconstrucción si hay al menos 10 stints comparables en esa carrera y todos coinciden; además, el stint faltante debe tener un único compuesto no desconocido en la fuente.

Con esa regla se recuperaron **390 filas de `Compound` en 17 stints**: 298 filas de Austria 2022, 35 de Canadá 2023 y 57 de Bélgica 2025. En las carreras correspondientes, la regla coincidió con todos los stints conocidos comparables: 44/44, 52/52 y 18/18, respectivamente. Los 35 compuestos faltantes de Mónaco 2022 siguen desconocidos porque la caché ofrece más de un compuesto posible para ese stint. La regla también rechaza carreras donde la alineación no es perfecta, como Australia 2025 (93/96), Bélgica 2022 (54/56) y Bélgica 2024 (49/53).

`timing_app_data` no contiene una medida directa de `TyreLife`; no se reconstruyó esa variable. Sus 633 faltantes elegibles siguen ausentes.

Las cifras elegibles excluyen filas generadas y filas sin stint o número de vuelta; por eso difieren del conteo bruto de nulos de la matriz completa. Las cadenas literales como `None` también deben normalizarse antes de contar faltantes, conservando la fuente original para auditoría.

## Decisión recomendada

- Mantener una sola fila canónica por piloto y vuelta.
- Reconstruir retrospectivamente solo huecos puntuales cuando existan vecinos válidos del mismo stint, dentro de la tolerancia, y la regla pase una validación equivalente.
- Dejar desconocidos los valores de stints completamente faltantes cuando no haya evidencia directa y validada en la caché; no inferirlos desde otro stint.
- Mantener trazabilidad de observado/reconstruido en el informe de auditoría, no como una segunda fila ni como predictor por defecto.
- Para una predicción histórica, permitir únicamente información disponible hasta el instante de predicción. Un vecino posterior puede ayudar a auditar, pero no puede alimentar esa predicción.

La matriz maestra permanece sin cambios. El script `EDA/dataset/reconstruir_compuestos_cache.py` genera una matriz candidata separada, conserva el mismo número de filas y columnas, y produce un CSV de auditoría con el valor y la fuente por stint. Los artefactos locales son `vueltas_2022_2025_compound_reconstruido.pkl` y `auditoria_reconstruccion_compound.csv` dentro de la carpeta de datos consolidados. La fuente puede reproducirse ejecutando el script; no necesita llamadas a la API.

## Relación con las especificaciones

- [Reglas candidatas de limpieza](../specs/04_limpieza_candidata.md): reglas generales y límites temporales.
- [Compuestos desconocidos y Austria 2022](../aclaraciones/01_compuestos_desconocidos.md): hallazgo previo en las carreras 2022–2023 y decisión sobre conservar Austria.
- [Script reproducible de reconstrucción](../../EDA/dataset/reconstruir_compuestos_cache.py): crea la matriz candidata y el informe de auditoría sin sobrescribir la matriz maestra.
