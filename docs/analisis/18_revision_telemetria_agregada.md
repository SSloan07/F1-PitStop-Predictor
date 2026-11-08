# Revisión 5: telemetría agregada por vuelta

**Fuente:** matriz `vueltas_2022_2025_limpieza_v1.pkl`, corte 2022–2023 de 47.997 filas y lógica en `EDA/dataset/consolidar_carreras.py`. No se modificaron datos ni hubo llamadas API. Revisa las columnas 45–61 del [inventario](13_inventario_columnas_para_modelado.md).

**Estado:** medias/máximos de velocidad y RPM, acelerador filtrado y Brake_mean acordadas como candidatas del baseline, bajo las reglas de calidad y tiempo descritas. Estas columnas resumen una vuelta completa; solo pueden alimentar la RNN como historial terminado y disponible. La matriz no contiene telemetría instantánea para la vuelta actual.

## Cómo se construyen los agregados

El extractor obtiene las muestras de coche de cada vuelta y calcula media y máximo sobre las muestras, además del número de muestras. Los promedios no ponderan por duración. Para acelerador calcula también variantes filtradas al rango 0–100; las variantes brutas conservan el código 104. No se filtran de forma equivalente marcha ni DRS.

FastF1 describe 104 como un código que puede indicar acelerador no disponible. Su documentación indica que DRS usa códigos parcialmente no descifrados y no recomienda tratarlos como magnitudes. Al implementar hay que fijar la versión de FastF1 y cotejar su diccionario [oficial](https://raw.githubusercontent.com/theOehrly/Fast-F1/main/fastf1/_api.py).

## Hallazgos en 2022–2023

| Columna/grupo | Hallazgo | Implicación |
|---|---|---|
| Telemetry_Samples | Mediana 340; 80 filas con más de 1.000 muestras, 25 con más de 5.000, máximo 14.910. | Cobertura/auditoría, no señal de ritmo. Algunos conteos extremos corresponden a vueltas con LapTime ausente o generadas. |
| Speed mean/max | 27 filas tienen ambos valores cero. | Revisar inmovilidad, boxes y bandera roja; no convertir cero a NA automáticamente. |
| RPM mean/max | 28 filas tienen ambos valores cero. | Mismo análisis de cobertura y estado. |
| Throttle bruto | 29 medias y 427 máximos superan 100; el máximo 104 aparece 427 veces. | Excluir los agregados brutos. |
| Throttle válido | 427 filas tienen fracción válida menor que 1; 27 no tienen muestras válidas y media/máximo son NA. | Son candidatos mejores; conservar cobertura y máscara. |
| Brake | Brake_max es True en 47.960 de 47.997 filas; la media va de 0 a 1. | El máximo casi no discrimina; la media es proporción de muestras, no duración de frenado. |
| Marcha | 8 medias exceden 8 (máximo 39,75); 29 máximos exceden 8 (máximo 128). | No usar agregados actuales. Filtrar muestras válidas 1–8 y recalcular desde origen; nunca recortar sin registro. |
| DRS | Máximos observados 0,1,2,3,8–15; media hasta 9,3. | Códigos de estado; no usar media/máximo como continuo hasta validar su semántica. |

## Decisión provisional por columna

| Columnas | Propuesta | Condición |
|---|---|---|
| Telemetry_Samples | Solo auditoría. | Reportar cobertura, no introducir conteo en X inicialmente. |
| Telemetry_Speed_mean/max | **Acordadas como candidatas RNN.** | Solo vuelta cerrada, máscara de validez; comparar ambas y revisar ceros/extremos. |
| Telemetry_RPM_mean/max | **Acordadas como candidatas RNN.** | Solo vuelta cerrada; revisar ceros y trazas anómalas. |
| Telemetry_Throttle_mean/max | Excluir variantes brutas. | Contienen valores 104. |
| Telemetry_Throttle_valid_mean/max | **Acordadas como candidatas RNN del baseline.** | Solo vuelta cerrada; conservar NA cuando no hay muestra válida y auditar cobertura. |
| Telemetry_Throttle_valid_samples/fraction | Solo auditoría al inicio. | Evitar que la cobertura del canal actúe como predictor indirecto de eventos o calidad retrospectiva. |
| Telemetry_Brake_mean | **Acordada como candidata RNN.** | Interpretarla como fracción de muestras con freno. |
| Telemetry_Brake_max | Excluir del baseline. | Casi constante y redundante con la media. |
| Telemetry_nGear_mean/max | Pausa hasta recalcular. | Validar marcha 1–8 en muestra cruda; no recortar agregados. |
| Telemetry_DRS_mean/max | Excluir por ahora. | La fuente oficial reconoce códigos sin decodificación completa. |

Todos los agregados de N quedan fuera de la decisión al inicio de N. Las 79 filas FastF1Generated requieren máscara/regla de elegibilidad; no se descarta la secuencia completa del piloto por una fila generada. Speed, RPM, acelerador y freno describen conducción, pero no miden directamente degradación del neumático.

## Comprobaciones antes de habilitarlas

1. Recalcular una muestra de agregados desde telemetría cruda y comparar contra la matriz, con versión FastF1 registrada.
2. Auditar el filtro de acelerador 0–100 y marcha 1–8, incluyendo casos sin muestras válidas.
3. Mantener DRS fuera hasta contar con una representación con significado sustentado.
4. Alterar datos posteriores a la decisión no debe cambiar entradas previas; ningún agregado de vuelta completa aparece en su token actual.
5. Comparar grupos por ablación; no asumir que más señales implican mejor predicción ni atribuir desgaste directo a estos resúmenes.

**Candidatas acordadas para el baseline:** Speed mean/max, RPM mean/max, acelerador filtrado y Brake_mean de vueltas cerradas no generadas; conteos en auditoría; DRS y marcha en pausa hasta validación/recalculo. La matriz aún no está aprobada para entrenamiento.
