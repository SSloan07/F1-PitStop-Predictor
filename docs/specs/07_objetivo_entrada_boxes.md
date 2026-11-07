# SPEC-LABEL-002: visita a boxes en las próximas tres vueltas

**Versión:** 1.1  
**Estado:** definición operativa acordada; las etiquetas existentes deben regenerarse y auditarse con esta regla.  
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

Las etiquetas candidatas existentes de 2022–2023 se produjeron con reglas anteriores y no cumplen necesariamente esta versión. Antes de entrenar se generará una versión nueva, conservando las anteriores para comparación. El conjunto 2024–2025 queda fuera de esa ejecución histórica; la disponibilidad de ambos registros no certifica por sí sola la exhaustividad de FastF1.

## Reproducibilidad y artefactos

La ejecución histórica v1 partió de `EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_limpieza_v1.pkl` y el registro de revisión `etiquetado_candidate_v1/casos_revision_usuario.csv`. La nueva ejecución deberá declarar las versiones exactas de sus entradas y de esta especificación.

Comando de la ejecución histórica v1, **no** de la nueva versión pendiente:

```bash
python scripts/etiquetar_paradas.py
```

Esa ejecución creó `etiquetado_pit_entry_v1/`. La nueva salida tendrá otro nombre y no sobrescribirá artefactos anteriores. Deberá guardar eventos, parejas, registros inciertos, decisiones de auditoría, etiquetas nullable, cobertura, manifiesto con SHA-256 y reporte. El etiquetado podrá repetirse localmente sin consultar la web.

La matriz original y el inventario anterior `candidate_v1` se conservan. El entrenamiento y las etiquetas definitivas quedan fuera de alcance.
