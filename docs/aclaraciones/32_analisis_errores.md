# Análisis de errores de los mejores modelos en validación 2024

**Estado:** completado sobre predicciones congeladas. No se reentrena ni se reelige ningún umbral. La prueba 2025 no se cargó.

Modelos analizados, con los umbrales F2 ya elegidos en el [ensamble](31_ensamble_gbm_lstm.md):

- **LightGBM, media de 3 semillas** (umbral 0,15): F2 0,535, precisión 28,5 %, recall 68,6 %, 3.993 FP, 726 FN.
- **Ensamble 50/50** (umbral 0,20): F2 0,529, precisión 25,0 %, recall 73,5 %, 5.113 FP, 613 FN.

Las cifras siguientes son de LightGBM salvo que se indique lo contrario.

```bash
python scripts/analisis_errores.py --output docs/aclaraciones/evidencia_errores_v1
```

## 1. Falsas alertas: ¿casi aciertos o alarmas puras?

Para cada falsa alerta en la vuelta N se midió cuántas vueltas faltaban para la siguiente entrada real a boxes del piloto. La etiqueta solo cuenta como acierto una entrada en N, N+1 o N+2.

| Siguiente entrada real | Falsas alertas | % |
|---|---:|---:|
| N+3 | 431 | 10,8 % |
| N+4 | 376 | 9,4 % |
| N+5 a N+6 | 615 | 15,4 % |
| N+7 a N+10 | 804 | 20,1 % |
| N+11 o más | 1.030 | 25,8 % |
| No vuelve a parar | 737 | 18,5 % |

- **Solo el 20 % son casi aciertos** (la parada llega 1–2 vueltas después del horizonte); el 36 % cae dentro de N+6.
- **Casi la mitad son alarmas puras:** la parada está a 11 vueltas o más (26 %), o el piloto no vuelve a parar (18 %). El modelo espera una parada por la edad del neumático y el piloto estira el stint o hace una parada menos.
- Solo 131 falsas alertas (3 %) ocurren en las dos vueltas siguientes a una parada propia.

**Métrica con tolerancia (complementaria):** si las alertas con la parada en N+3 o N+4 se cuentan como anticipadas en vez de falsas, la precisión pasa de 28,5 % a 33,3 % y el **F2 de 0,535 a 0,566** (ensamble: 0,529 → 0,560). La etiqueta de 3 vueltas explica parte de la distancia a la meta F2 ≥ 0,60, pero no toda.

## 2. Paradas no anticipadas

LightGBM cubre 640 de 824 paradas (77,7 %). La cobertura depende mucho del tipo de parada:

| Tipo de parada (evento) | Eventos | Cubiertos | Cobertura |
|---|---:|---:|---:|
| En verde, stint normal o largo | 592 | 510 | **86,1 %** |
| Bajo SC/VSC, stint normal o largo | 96 | 77 | 80,2 % |
| Bajo SC/VSC, stint corto (< 50 % del típico) | 75 | 34 | 45,3 % |
| En verde, stint corto (< 50 % del típico) | 61 | 19 | **31,1 %** |
| Primera parada | 457 | 380 | 83,2 % |
| Parada intermedia | 92 | 59 | 64,1 % |
| Última de varias paradas | 275 | 201 | 73,1 % |

(Ensamble: 87,8 % en stints normales y 50,0 % en stints cortos.)

- **El problema principal son las paradas tempranas**, cuando el stint dura menos de la mitad de lo típico para su compuesto: 136 eventos, solo 39 % cubiertos. Las 184 paradas no anticipadas tienen una mediana de 9 vueltas de neumático, frente a 20 en las cubiertas. Son paradas oportunistas bajo SC, por daño, por pinchazo o por cambio de clima, sin señal previa en el desgaste.
- **El Safety Car por sí solo no es el problema:** si el stint ya era normal, se cubre el 80 % de las paradas bajo SC/VSC.
- Las paradas intermedias y la última de una estrategia de varias se anticipan peor que la primera.
- **Anticipación:** de las 640 cubiertas, 505 tienen alerta con 2 vueltas de antelación, la máxima que permite el horizonte; 64 con 1 vuelta y 71 solo en la misma vuelta.

## 3. Dónde falla: carreras

El F2 por carrera va de 0,16 a 0,79.

| Peores | F2 | Eventos cubiertos | | Mejores | F2 | Eventos cubiertos |
|---|---:|---:|---|---|---:|---:|
| Arabia Saudí | 0,16 | 5/19 | | Baréin | 0,79 | 42/43 |
| Mónaco | 0,24 | 10/23 | | Australia | 0,73 | 34/36 |
| Qatar | 0,32 | 41/60 | | España | 0,72 | 41/42 |
| Emilia-Romaña | 0,36 | 17/27 | | São Paulo | 0,71 | 32/34 |

Las peores son carreras atípicas: paradas en masa bajo Safety Car muy temprano o cambios de neumático durante una bandera roja (Arabia Saudí, Mónaco), y paradas forzadas o muy frecuentes (Qatar). Las mejores son carreras de degradación clásica, donde la parada sigue al desgaste. Promediar sobre carreras tan distintas esconde que el modelo funciona bien en las "normales".

## Conclusiones

1. **El modelo ya anticipa bien las paradas planificadas:** 86 % de cobertura en stints normales en verde, casi siempre con dos vueltas de antelación.
2. **El techo del F2 por muestra viene de dos fuentes:**
   - paradas tempranas imprevisibles con la información disponible (≈17 % de los eventos, 39 % cubiertas);
   - alarmas puras cuando el piloto estira el stint o hace una parada menos (≈45 % de las falsas alertas).
3. La etiqueta de 3 vueltas aporta algo (F2 tolerante 0,566), pero no es la causa principal.

## Qué se puede hacer

- **Reducir las alarmas puras:** variables sobre la estrategia de paradas restantes, por ejemplo cuántas paradas suele haber en esa carrera (prior de train por circuito) o si con el neumático actual se puede llegar al final con margen. Hasta ahora `laps_short_to_finish` no ayudó en LightGBM. Habría que revisar si lo que falta es saber si la estrategia es de una o de dos paradas.
- **Reportar métricas por tipo de parada** (planificada en verde frente a temprana u oportunista) y la métrica con tolerancia como complemento, para describir el rendimiento con honestidad.
- **No perseguir las paradas por daño o pinchazo:** no hay señal causal previa en los datos.
- Para la evaluación final en 2025, decidir de antemano qué métricas se reportan (F2 por muestra como principal, más cobertura por tipo de parada y F2 tolerante).
