# Experimento pareado: LSTM32 con entradas completas frente a solo telemetría

**Estado:** preparado para Colab CPU. El test 2025 permanece reservado.

## Pregunta

La mejor LSTM hasta ahora (LSTM32 de una capa, dropout 0,2, BCE `pos_weight=2`, selección por F2; F2 de validación 0,46223 con semilla 42, ver [resultados de tres semillas](../aclaraciones/25_resultados_pos_weight_tres_semillas.md)) recibe identidad del piloto, equipo y circuito, y clima casi constante dentro de cada carrera. ¿Mantiene su F2 si se le quitan esas variables y se deja principalmente la información que cambia vuelta a vuelta: ritmo, neumático, posición, estado de pista y telemetría del coche?

La hipótesis es doble: si el F2 se mantiene, el modelo no dependía de memorizar contexto; si baja, ese contexto aportaba información útil (por ejemplo, la estrategia típica de cada circuito).

## Entradas

Se deriva `EDA/dataset/muestras_preprocesadas_v1_3_telemetria_v1/` del paquete congelado `muestras_preprocesadas_v1_3` mediante una **proyección de columnas**. No se reajustan estadísticas: la estandarización y los vocabularios se ajustan por variable, así que las medias, escalas y vocabularios conservados son exactamente los que produciría un ajuste train-only sobre el subconjunto. El orden de muestras, las etiquetas, los metadatos, las máscaras por posición y la población elegible son idénticos (46.162 muestras de train y 25.635 de validación).

| Grupo | Completas (referencia) | Solo telemetría (candidata) |
|---|---|---|
| Progreso y neumático | `LapNumber`, `TyreLife_last_eligible` | Igual |
| Ritmo y posición | `LapTime`, `Sector1–3Time`, `Position` | Igual |
| Telemetría del coche | Velocidad media/máx., RPM media/máx., acelerador medio/máx., freno medio | Igual |
| Clima | 8 variables | Solo `Weather_Rainfall` |
| Categóricas dinámicas | `Compound_asof`, `track_phase_asof` | Igual |
| Contexto estático | `Driver`, `Team`, `GranPremio` | Ninguno (marcador constante) |

Se eliminan `Driver`, `Team`, `GranPremio`, `Weather_AirTemp`, `Weather_Humidity`, `Weather_Pressure`, `Weather_TrackTemp`, `Weather_WindDirection_sin`, `Weather_WindDirection_cos` y `Weather_WindSpeed`.

**Criterio.** Sobre las celdas presentes de train elegible se calculó qué parte de la varianza de cada variable ocurre dentro de una misma carrera (el resto distingue unas carreras de otras). Los valores se publican en `report.json` del paquete:

| Variable | Varianza dentro de carrera |
|---|---:|
| `Weather_Pressure` | 0,0 % |
| `Weather_AirTemp` | 0,6 % |
| `Weather_Humidity` | 1,7 % |
| `Weather_TrackTemp` | 4,3 % |
| `Weather_WindSpeed` | 24,5 % |
| `Weather_WindDirection_sin/cos` | 46 % |
| `Weather_Rainfall` | 62,7 % |
| `LapTime` | 49,9 % |
| `TyreLife_last_eligible` | 90,2 % |

Temperatura, humedad y presión son prácticamente constantes por carrera y funcionan como identificador del circuito y la fecha. El viento es ambiental y no describe el estado del coche. La lluvia sí cambia dentro de la carrera y provoca cambios a intermedios, por eso se conserva. `Driver`, `Team` y `GranPremio` son constantes por definición para cada piloto-carrera.

**Rama estática.** Para no cambiar la arquitectura, la rama FFNN estática se mantiene, pero recibe una sola columna `SinContextoEstatico` con el ID reservado 1 (ausencia) en todas las muestras. Su salida es un vector constante equivalente a un sesgo. El número de parámetros baja de 12.377 a 10.189 porque disminuyen la entrada de la LSTM y las tablas de embeddings, no por un cambio de capas.

Reproducción del paquete (rechaza la carpeta si ya existe):

```bash
python scripts/preparar_entradas_telemetria.py --config configs/entradas_telemetria_v1.json
```

## Diseño ortogonal

| Control | Referencia | Candidata |
|---|---|---|
| Entradas | `muestras_preprocesadas_v1_3` | `muestras_preprocesadas_v1_3_telemetria_v1` |
| Arquitectura | LSTM32 de una capa, FFNN estática 16, cabeza 16, dropout 0,2 | Igual |
| Pérdida / optimizador / tasa | BCE `pos_weight=2` / Adam / 0,001 fija | Igual |
| Selección | F2 a 0,5 para checkpoint; F2 en la cuadrícula congelada para el umbral | Igual |
| Train / validación / test | 2022–2023 / 2024 / 2025 reservada | Igual |
| Semilla / entorno | 42 / Colab CPU, misma sesión | Igual |

`scripts/entrenar_comparacion_telemetria.py` valida que las configuraciones solo difieran en `preprocessed_package` y su hash, que la candidata declare ser una proyección del paquete de referencia, que ambos hijos usen el mismo código y la misma población, y que las especificaciones de modelo solo difieran en dimensiones de entrada. Configuraciones: `configs/experimento_lstm32_telemetria_ref_cpu_v1.json` y `configs/experimento_lstm32_telemetria_candidate_cpu_v1.json`.

La referencia repite la receta de `lstm32_pos_weight2_f2_cpu_v1_seed42/lstm32_pos_weight2` en la misma sesión, para que la comparación no dependa de diferencias de entorno con la corrida anterior.

## Ejecución

1. Desde la raíz: `python scripts/empaquetar_telemetria.py --output runs/lstm_colab_telemetria_cpu_v1.zip`. El ZIP publicado y su SHA-256 están fijados en el notebook.
2. Abrir `notebooks/12_lstm32_telemetria_f2_colab_cpu.ipynb` en Colab, subir el ZIP y ejecutar todas las celdas.
3. Devolver `lstm32_telemetria_f2_cpu_v1_seed42.zip` intacto a `runs/`. Para la evaluación local de la candidata, usar `python scripts/evaluar_rnn.py --bundle <hijo> --data EDA/dataset/muestras_preprocesadas_v1_3_telemetria_v1 --output <nueva carpeta>`.

## Prueba local preliminar (no oficial)

Para validar el flujo completo, la pareja se ejecutó una vez en CPU local (Linux, PyTorch 2.x CPU, Python 3.12), con el mismo notebook simulando Colab, y se evaluó con `scripts/evaluar_rnn.py` (paridad exacta, cero decisiones distintas). El entorno difiere de Colab: la referencia obtuvo F2 0,45536 en lugar del 0,46223 de la corrida Colab anterior, así que estos números solo sirven como orientación. La corrida oficial es la de Colab.

| Validación 2024 | Completas | Solo telemetría | Diferencia |
|---|---:|---:|---:|
| Época / umbral | 15 / 0,17 | 14 / 0,21 | — |
| F2 | 0,45536 | 0,45885 | +0,00349 |
| F1 | 0,28789 | 0,30270 | +0,01481 |
| Precisión | 17,85 % | 19,32 % | +1,47 pp |
| Recall por muestra | 74,38 % | 69,94 % | −4,45 pp |
| AP | 0,21117 | 0,21159 | +0,00041 |
| FP / FN | 7.926 / 593 | 6.763 / 696 | −1.163 / +103 |
| Eventos cubiertos | 662/824 | 614/824 | −48 |
| Parámetros | 12.377 | 10.189 | −2.188 |

Sin contexto estático ni clima constante, el F2 y el AP se mantienen prácticamente iguales: el modelo no dependía de esas variables para su puntaje. Sí cambia el punto de operación: menos falsas alertas, a cambio de menor recall y 48 eventos menos cubiertos. La diferencia de F2 es menor que la variación entre semillas.

Es una sola semilla. Si la diferencia de F2 es menor que la variación observada entre semillas (≈0,017 en la pareja `pos_weight`), se replicará con 43 y 44 antes de concluir.
