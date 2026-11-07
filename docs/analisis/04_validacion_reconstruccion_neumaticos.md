# Validación de reconstrucción de datos de neumáticos

**Estado:** evaluación retrospectiva; no se modificó la matriz.
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

Las cifras elegibles excluyen filas generadas y filas sin stint o número de vuelta; por eso difieren del conteo bruto de nulos de la matriz completa. Las cadenas literales como `None` también deben normalizarse antes de contar faltantes, conservando la fuente original para auditoría.

## Decisión recomendada

- Mantener una sola fila canónica por piloto y vuelta.
- Reconstruir retrospectivamente solo huecos puntuales cuando existan vecinos válidos del mismo stint, dentro de la tolerancia, y la regla pase una validación equivalente.
- Dejar desconocidos los valores de stints completamente faltantes; no inferirlos desde otro stint.
- Mantener trazabilidad de observado/reconstruido en el informe de auditoría, no como una segunda fila ni como predictor por defecto.
- Para una predicción histórica, permitir únicamente información disponible hasta el instante de predicción. Un vecino posterior puede ayudar a auditar, pero no puede alimentar esa predicción.

La matriz permanece sin cambios. Antes de ejecutar una reconstrucción, conviene comprobar si las fuentes o cachés originales recuperan esos stints; si no, los faltantes se conservarán y el tratamiento se resolverá en el pipeline del experimento.

## Relación con las especificaciones

- [Reglas candidatas de limpieza](../specs/04_limpieza_candidata.md): reglas generales y límites temporales.
- [Compuestos desconocidos y Austria 2022](../aclaraciones/01_compuestos_desconocidos.md): hallazgo previo en las carreras 2022–2023 y decisión sobre conservar Austria.
