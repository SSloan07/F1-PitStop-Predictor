# Aclaración: telemetría anómala en Japón y Miami 2022

**Estado:** investigación y propuesta; no se han modificado los datos.

En Japón 2022, BOT presenta 26 vueltas consecutivas (3–28) con velocidad y RPM medias cero y acelerador medio 104. En Miami 2022, MAG presenta el mismo patrón en la vuelta 1; en la siguiente se observan valores de velocidad y RPM distintos de cero. Para dos vueltas de BOT y la vuelta afectada de MAG se cotejaron los valores con las muestras originales dentro de los tiempos de vuelta. El patrón está presente en esas muestras, pero su causa sigue sin determinarse.

La propuesta es conservar las filas y la fuente original, señalar los intervalos afectados y tratar los resúmenes afectados como faltantes en una copia procesada, una vez aprobadas reglas y pruebas. No reemplazar con telemetría de vueltas futuras. La falta de un canal no implica que todos los demás estén inválidos. Falta revisar si existen otros patrones anómalos que no produzcan velocidad media exactamente cero y establecer criterios generales.

Detalle y método: `docs/analisis/02_calidad_inicial_entrenamiento.md`.

## Búsqueda ampliada

En 2022–2023 hay 427 resúmenes con acelerador máximo mayor a 100, 29 con acelerador medio mayor a 100 y 73 con velocidad media menor a 10. Esas condiciones no son equivalentes a un canal totalmente inválido: una vuelta lenta puede incluir interrupciones, y un máximo puede representar pocas muestras. Se revisaron además tres ejemplos en la caché original, dos de ellos vueltas generadas con tramos extensos de acelerador 104. Ver cifras y limitaciones en el informe de calidad. La causa y el significado de 104 siguen pendientes; no se ha aplicado una corrección.
