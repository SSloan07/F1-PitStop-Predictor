# Simulador interactivo de modelos · validación 2024

**Estado:** MVP local implementado sobre los mockups de `Frontend/`. La matriz histórica se abre como fuente común y el replay filtra únicamente 2024. Los datos/predicciones de 2025 no participan ni se muestran; quedan reservados para la evaluación final.

El usuario puede cambiar entre ocho artefactos preexistentes y reproducir cualquiera de las 24 carreras de validación. La API devuelve probabilidades guardadas por los checkpoints, ordenadas por el `decision_time` de cada muestra. El frontend muestra las posiciones observadas hasta el último cierre de vuelta conocido, riesgos sobre el umbral individual del modelo y paradas reales solo cuando llega su `entry_time`.

La predicción significa entrada a boxes entre la vuelta N y N+2. Se generan alertas por piloto y puede haber más de un piloto por instante. El tablero muestra objetivo de selección, arquitectura, umbral y métricas del artefacto seleccionado. Las cifras 2024 son de desarrollo y no prueban generalización.

La app no recalcula pesos durante el replay: reproduce predicciones de validación exportadas en los ZIPs. Es una simulación histórica por tiempo de sesión, no serving de un feed de carrera en vivo. Los resultados están alineados al mismo split, población, preprocesador y horizonte. El backend verifica hashes de las fuentes y artefactos al arrancar y no expone `target` ni etiquetas futuras.

Ejecución: `python3.13 scripts/servir_simulador_2024.py`, luego `http://127.0.0.1:8765`. Instrucciones completas: [`Frontend/README.md`](../../Frontend/README.md).

La repetición de carrera permite comparar comportamiento visual. No se deben ajustar modelos o umbrales usando 2024 otra vez sin registrar el cambio; 2025 queda reservada para una evaluación final después de congelar las decisiones.


## Visualización de los autos y paradas

La vista de pista consume `position_data.ff1pkl` (X/Y y fecha de muestra) y `car_data.ff1pkl` (velocidad) de la caché FastF1 local de la carrera seleccionada. Las cachés se cargan bajo demanda, una carrera a la vez. La pista se reconstruye con una vuelta completa observada; la línea discontinua marca el tramo de pit lane derivado de una parada real.

El halo rojo identifica alertas desde su `decision_time`. La entrada real se revela al llegar a `entry_time`; el marcador pasa a ámbar dentro del intervalo PitIn/PitOut y a verde durante la detención observada (velocidad ≤2 km/h). El registro diferencia alerta, entrada y detención para que se vea cuánto anticipó el modelo. El historial de posiciones mostrado por vuelta sigue usando vueltas cerradas. La animación suaviza visualmente posiciones entre muestras ya recibidas y no altera las predicciones ni su cronología.

Esta vista requiere cachés locales FastF1 para las 24 carreras 2024. No descarga datos de la API y no accede a 2025. Los archivos de caché siguen fuera de Git.
