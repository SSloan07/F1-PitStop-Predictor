# Aclaración: compuestos desconocidos y GP de Austria 2022

**Estado:** hallazgo y decisión pendiente; no se ha eliminado ninguna carrera.  
**Alcance:** archivos de carrera 2022–2023 disponibles en `EDA/dataset/datos_carreras_completos_2022_2025`.

## Hallazgo

Hay 369 filas con el compuesto registrado como la cadena literal `None`: 35 en Mónaco 2022, 299 en Austria 2022 y 35 en Canadá 2023. En la exploración ninguna tenía un compuesto conocido en otra vuelta del mismo piloto, carrera y stint. El texto `None` no es un nulo de pandas; su significado debe normalizarse explícitamente y conservar el dato original.

Austria 2022 contiene 1.324 filas, de las cuales 299 tienen compuesto desconocido; afectan a 12 de 20 pilotos. Toda la carrera representa aproximadamente el 2,76 % de las 47.997 filas examinadas de 2022–2023 y contiene 44 registros de entrada a boxes. Estos son eventos de entrada, no necesariamente cambios de neumáticos confirmados. La carrera también aporta clima, telemetría y otros datos que podrían ser utilizables.

## Decisión por acordar

Opciones: conservar la carrera e indicar valores desconocidos; excluirla de ciertos experimentos o de todas las muestras de entrenamiento; o contrastar con una fuente independiente antes de decidir. Excluirla completa también descarta las 1.025 filas con compuesto conocido y los demás datos disponibles. No se ha comprobado aún si recuperar el compuesto por fuente externa sería válido para el instante de predicción.

**Propuesta:** conservarla en la matriz maestra; no aplicar una exclusión definitiva antes de investigar y cuantificar su impacto. Una comparación posterior con y sin la carrera podría mostrar sensibilidad, registrando los cambios y evaluando en la misma validación. Ninguna regla de selección o imputación debe usar datos futuros como entrada del modelo.

## Referencias internas

- `docs/analisis/02_calidad_inicial_entrenamiento.md`: diagnóstico y búsqueda de compuestos conocidos en el mismo stint.
- `docs/specs/04_limpieza_candidata.md`: propuesta de tratamiento de valores desconocidos.
