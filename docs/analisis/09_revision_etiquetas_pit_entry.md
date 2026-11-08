# Revisión inicial de eventos y etiquetas de visita a boxes

**Ejecución:** `etiquetado_pit_entry_v1`  
**Temporadas:** 2022–2023  
**Selección de ejemplos:** semilla 42; cinco positivas y cinco negativas.

## Eventos sin resolver

El inventario contiene 1.779 eventos anclados en `PitInTime`: 1.715 clasificados como visitas a boxes por pareja temporal `PitInTime`/`PitOutTime` o confirmación FIA, y 64 desconocidos.

| Motivo | Casos | Observación |
|---|---:|---|
| `pit_in_without_confirming_pit_out` | 60 | Hay `PitInTime`, pero no se encontró una salida posterior emparejable. |
| `FIA_does_not_list_candidate_stop` | 4 | El candidato no figura como parada en el resumen FIA revisado. |

De los 64 desconocidos, 63 están en la última vuelta disponible del piloto y uno está una vuelta antes. Los 60 casos sin salida tampoco tienen una fila posterior para observar un nuevo stint. El patrón es compatible con retiros o final de datos, pero no basta para adjudicar un motivo; se conservan como desconocidos. Los cuatro candidatos que no aparecen en FIA son Verstappen en Baréin v.54, Ricciardo y Bottas en Arabia Saudita v.36 y Alonso en Imola v.6.

También hay 49 `PitOutTime` sin entrada emparejada; todos corresponden a la vuelta 1, consistente con la liberación de la parrilla. Se registran en `orphan_exits.csv` y no se consideran paradas de carrera.

## Ejemplos de etiquetas

### Positivas

En cada caso existe una visita clasificada dentro del horizonte inclusivo `{N, N+1, N+2}`. Los tres números de vuelta de la ventana estaban presentes y no generados.

| GP | Temporada | Piloto | Vuelta de muestra N | Vuelta del evento | Desfase |
|---|---:|---|---:|---:|---:|
| Estados Unidos | 2023 | Hülkenberg | 32 | 34 | N+2 |
| Francia | 2022 | Bottas | 16 | 18 | N+2 |
| São Paulo | 2022 | Schumacher | 45 | 46 | N+1 |
| Austria | 2022 | Gasly | 40 | 40 | N |
| Gran Bretaña | 2023 | Hülkenberg | 5 | 7 | N+2 |

### Negativas

En estos casos las vueltas `N`, `N+1` y `N+2` están presentes, no generadas y no contienen un evento clasificado.

| GP | Temporada | Piloto | Vuelta de muestra N | Horizonte examinado |
|---|---:|---|---:|---|
| Ciudad de México | 2022 | Latifi | 32 | 32–34 |
| Canadá | 2023 | Tsunoda | 25 | 25–27 |
| Países Bajos | 2023 | Lawson | 29 | 29–31 |
| Bélgica | 2023 | Ricciardo | 12 | 12–14 |
| São Paulo | 2022 | Albon | 8 | 8–10 |

## Recuento de etiquetas

El artefacto tiene 47.997 muestras: 4.817 positivas, 41.361 negativas y 1.819 desconocidas. El dtype de `PitWithin3Laps` es `boolean` nullable.

Las etiquetas negativas son negativas respecto a los eventos observables en la matriz y a la cobertura definida; no son una verificación física independiente. Las positivas por parejas de telemetría usan la definición operacional aprobada de visita a boxes, no afirman que se hayan cambiado neumáticos ni especifican el motivo.

## Próxima revisión

Antes de entrenar, revisar de forma dirigida los 60 `PitInTime` sin `PitOutTime`, comenzando por los casos que no están en la última vuelta disponible y por los cuatro candidatos con discrepancia FIA. No convertirlos a positivo o negativo sin evidencia que resuelva si hubo visita efectiva. Después, revisar más ejemplos por piloto/carrera y confirmar que la política de cobertura de etiquetas negativas es adecuada.
