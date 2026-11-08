# SPEC-LABEL-002: visita a boxes en las próximas tres vueltas

**Versión:** 1.1  
**Estado:** definición operativa acordada; etiquetas v3 publicadas para 2022–2025.
**Decisión:** la etiqueta predice una visita a boxes con `PitInTime` y `PitOutTime` emparejados, sin exigir evidencia de cambio de neumáticos.

## Objetivo

Para cada piloto y vuelta `N`, predecir si se registrará una visita a boxes en `N`, `N+1` o `N+2`. La predicción corresponde al inicio de la vuelta. El horizonte usa números de vuelta, no las tres filas siguientes.

`PitInTime` inicia el evento y determina la vuelta. Solo se cuenta cuando existe un `PitOutTime` posterior emparejado del mismo piloto y carrera; la salida no genera un segundo evento. Una entrada sin salida o una salida sin entrada no es positiva ni negativa por sí sola: queda como evento incierto para los horizontes afectados. La salida de la vuelta 1 correspondiente a la largada se ignora.

El objetivo es **visita a boxes**, no cambio de neumáticos. Se cuentan entradas y salidas emparejadas durante la carrera bajo condiciones normales, Safety Car o Virtual Safety Car; la neutralización es contexto y no sustituye la evidencia de la pareja. Una visita por penalización o reparación cumple la regla si tiene ambos registros. Un traslado del coche por comisarios no equivale a una entrada realizada por el piloto. Una entrada sin salida asociada a abandono queda incierta, incluso cuando una fuente oficial confirme el ingreso; esa fuente se conserva para auditoría, pero no reemplaza el requisito de `PitOutTime`. Los casos generados y conflictos de evidencia requieren revisión.

## Etiquetado

- `True`: existe al menos una pareja válida `PitInTime`/`PitOutTime` cuya entrada ocurrió en `N`, `N+1` o `N+2`, después del instante de decisión.
- `False`: las tres vueltas están observadas, no generadas, tienen tiempos coherentes y no existe evento positivo ni evento sin resolver en ese horizonte.
- `pd.NA`: muestra actual no válida, horizonte incompleto o evento sin resolver.

Los tiempos se emparejan en orden dentro de la misma sesión; ninguna salida puede reutilizarse ni asociarse cruzando una nueva entrada. La evidencia FIA y otras fuentes confiables sirven para auditar discrepancias, no para clasificar como positivo un evento sin la pareja exigida. Tampoco se convierte en negativo una entrada incompleta porque falte en un resumen oficial. Un `PitOutTime` huérfano no permite asignar una vuelta de entrada. El momento de la entrada debe ser posterior a la decisión para contarla como evento futuro.

Las etiquetas candidatas anteriores se conservan para trazabilidad. La versión vigente `etiquetado_pit_entry_v3_2022_2025` cubre 2022–2025 con esta regla. Su publicación acredita consistencia con las parejas disponibles en la matriz fuente, pero no certifica por sí sola la exhaustividad de FastF1.

## Reproducibilidad y artefactos

La ejecución histórica v1 partió de `EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_limpieza_v1.pkl`; no cumple necesariamente este contrato y no debe tratarse como la versión vigente. El código preparado para la próxima ejecución usa la matriz maestra consolidada `vueltas_2022_2025.pkl` y el registro de revisión `etiquetado_candidate_v1/casos_revision_usuario.csv`, conservando ambos como entradas solo lectura. El registro externo sirve para revisar discrepancias, no para sustituir la pareja requerida. Cada nueva ejecución debe declarar los hashes exactos de sus entradas y esta especificación.

El directorio preexistente `etiquetado_pit_entry_v2/` tiene un manifiesto que lo identifica como `pit_entry_v1` y como fuente `vueltas_2022_2025_limpieza_v1.pkl`; su nombre no representa la regla vigente. Se conserva intacto y no se usa como salida aprobada.

Comando actual de generación de la candidata emparejada:

```bash
python scripts/etiquetar_paradas.py
```

Esa ejecución histórica creó `etiquetado_pit_entry_v1/`. La candidata `pit_entry_v2` ya se generó en `etiquetado_pit_entry_v2_pair_required/` desde la matriz maestra y el registro de revisión, ambos identificados por hash. Produjo **47.997** muestras: **4.817** positivas, **41.361** negativas y **1.819** desconocidas; detectó **1.715** parejas y dejó **64** eventos sin resolver. La verificación independiente confirmó que las 4.817 positivas tienen una pareja en el horizonte y entrada posterior al instante de decisión. El directorio preexistente con nombre `etiquetado_pit_entry_v2/` permanece sin cambios y no corresponde a esta ejecución.

El código rehúsa sobrescribir la salida si ya existe. Deja incierta cualquier entrada sin salida emparejada, incluso si una fuente externa la reporta; una salida huérfana queda en auditoría sin asignarle vuelta de entrada ni contaminar automáticamente un horizonte. La ejecución registra eventos, salidas huérfanas, registros inciertos, reclamos externos tipados por origen, etiquetas nullable, cobertura, hashes de entradas/código/especificación, manifiesto y reporte. El etiquetado puede repetirse localmente sin consultar la web.

## Publicación vigente 2022–2025

La versión `etiquetado_pit_entry_v3_2022_2025` se generó desde la matriz maestra `vueltas_2022_2025.pkl` (SHA-256 `9f8eb3f40d66869778c30d75f0a1ebc45196b8ad03c3a4258cc8f5e335dab65c`) con la regla `pit_entry_v2`, horizonte de tres vueltas y evidencia de pareja `PitInTime`/`PitOutTime`. Incluye 101.290 claves: 9.393 positivas, 88.163 negativas y 3.734 desconocidas. La comparación por clave de 2022–2023 con `etiquetado_pit_entry_v2_pair_required` fue idéntica en las 47.997 muestras (valor, estado y campos de elegibilidad).

Reproducción: `python scripts/etiquetar_paradas.py --config configs/etiquetado_pit_entry_v3_2022_2025.json`. La salida y su manifiesto están bajo `EDA/dataset/datos_carreras_completos_2022_2025/etiquetado_pit_entry_v3_2022_2025/`. El corte 2024–2025 se etiquetó con la misma regla; no se usaron sus distribuciones para cambiar el objetivo.

La matriz original y el inventario anterior `candidate_v1` se conservan. El entrenamiento y las etiquetas definitivas quedan fuera de alcance.
