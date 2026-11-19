# Modelo final congelado y protocolo de evaluación en 2025

**Estado:** modelo congelado; **2025 todavía no se ha cargado ni evaluado.** Este documento fija, antes de mirar 2025, qué se evalúa y cómo se reporta. Cualquier cambio posterior a este documento invalida la evaluación como prueba independiente y debe declararse.

## 1. Modelo congelado

| Componente | Valor |
|---|---|
| Artefactos | `modelos/lightgbm_final_v1/` (boosters LightGBM en texto, orden de variables, parámetros, umbral, predicciones de validación y manifiesto con hashes) |
| Algoritmo | LightGBM binario, media de las probabilidades de tres modelos (semillas 42, 43 y 44) |
| Entradas | 66 variables por decisión: resumen de la ventana causal de telemetría (último valor y media de 8 vueltas, compuesto y estado de pista), sin contexto estático ni clima constante, más las variables de estrategia de las rondas 1, 2 y 3 ([29](../aclaraciones/29_diagnostico_gradient_boosting.md), [30](../aclaraciones/30_segunda_ronda_variables_estrategia.md), [33](../aclaraciones/33_variables_numero_paradas.md)) |
| Hiperparámetros | `random_06` ([34](../aclaraciones/34_ajuste_hiperparametros_gbm.md)): 127 hojas, mín. 100 muestras por hoja, lr 0,01, `feature_fraction` 0,7, `bagging_fraction` 0,6, `lambda_l2` 5, `min_gain_to_split` 0,5, `max_bin` 63 |
| Entrenamiento | Train 2022–2023; número de árboles por validación cruzada por carrera en train |
| Umbral de alerta | **0,14**, elegido maximizando F2 en validación 2024 |
| Validación 2024 (desarrollo) | AP 0,3867, F2 0,5399, precisión 27,9 %, recall 70,5 % |

SHA-256 del manifiesto congelado: `e23908f3b6b0e6a4c077d53d2b2498aa9ee526a81744267ff9db56ea275f5f06`. Reproducción (CPU, unos 10 minutos; `requirements-diagnostico.txt`): `python scripts/congelar_modelo_final.py --output <carpeta nueva>`.

`scripts/congelar_modelo_final.py` reentrena los tres modelos y solo publica si reproducen exactamente los árboles, el umbral y las métricas documentadas. Después vuelve a cargar los ficheros guardados y comprueba que dan las mismas predicciones.

Valores de referencia aprendidos solo de train y usados también en 2025: duración típica de stint por compuesto, stint largo (percentil 90), número típico de paradas por GP y pérdida de tiempo por parada por GP. En 2025, la pérdida de tiempo por parada se actualiza con paradas anteriores de la misma carrera, igual que en train y validación (regla causal, nunca con datos posteriores a la decisión).

## 2. Qué se evaluará en 2025

**Población:** todas las muestras de 2025 con `split = test_final_reserved` y `supervised_eligible = true` según las particiones `muestras_causales_v1_3_etiquetas_v3_particiones_v1_1`, con etiquetas v3. Los mismos criterios que en train y validación.

**Métrica principal:** F2 de la clase positiva por muestra, al umbral congelado 0,14.

**Métricas secundarias**, todas reportadas aunque sean desfavorables:

1. AP, precisión, recall, F1, falsos positivos y falsos negativos al umbral 0,14.
2. Cobertura por evento (parada confirmada con al menos una alerta en su horizonte causal), en total y por tipo: en verde con stint normal, bajo SC/VSC con stint normal, y stint corto (< 50 % del típico), con las mismas definiciones del [análisis de errores](../aclaraciones/32_analisis_errores.md).
3. F2 con tolerancia: falsas alertas cuya siguiente parada llega en N+3 o N+4 se cuentan como anticipadas.
4. Distribución de F2 por carrera.
5. Como referencia, el mismo informe de validación 2024 al lado.

**No se hará en 2025:** ajustar el umbral, reentrenar, cambiar variables, parámetros ni semillas, ni excluir carreras. Si una carrera de 2025 no tiene datos de FastF1 para alguna fuente, sus muestras siguen la política de máscaras o ausencias ya vigente y se reportan, no se eliminan.

## 3. Pipeline de evaluación (listo, revisado y commiteado antes de ejecutar 2025)

`scripts/evaluar_modelo_final.py` implementa los tres pasos con un solo camino de código para 2024 y 2025:

1. **Comprobación del congelado:** el manifiesto debe coincidir con el SHA-256 registrado arriba, y los modelos y todos los ficheros fuente que lista (módulos de estrategia, scripts de diagnóstico, ajuste y congelado) deben ser idénticos byte a byte.
2. **Entradas base:** transforma las ventanas causales crudas de `muestras_causales_v1_3` con el preprocesador congelado de `muestras_preprocesadas_v1_3` (`transform`, sin ajuste) y toma población, elegibilidad y etiquetas de `..._particiones_v1_1`.
3. **Variables de estrategia:** las calcula con los módulos congelados. Los valores de referencia se aprenden solo de 2022–2023 y la distancia programada sale de la caché FastF1 de la temporada.
4. **Predicción y métricas:** media de los tres boosters guardados, umbral 0,14 y todas las métricas de la sección 2. Escribe predicciones, eventos, F2 por carrera y un manifiesto.

**Ensayo en 2024 (`--split validation`):** reconstruye desde cero las entradas de 2024 y exige que los arrays transformados sean idénticos al paquete publicado y que las probabilidades coincidan con las congeladas. Resultado: idénticas, diferencia máxima 9,9×10⁻¹⁷. El informe del ensayo, con el mismo formato que tendrá el de 2025, está en `docs/aclaraciones/evaluacion_final_ensayo_2024/`:

| Validación 2024 (referencia) | Valor |
|---|---:|
| F2 (principal) | 0,5399 |
| AP / precisión / recall | 0,3867 / 27,9 % / 70,5 % |
| Cobertura por evento | 650/824 (78,9 %) |
| En verde con stint normal / SC-VSC con stint normal / stint corto | 86,1 % / 78,1 % / 47,8 % |
| F2 con tolerancia N+3/N+4 | 0,5713 |
| F2 por carrera (mín. / mediana / máx.) | 0,19 / 0,55 / 0,80 |

**Ejecución única en 2025:**

```bash
python scripts/evaluar_modelo_final.py --split test --confirmar-evaluacion-unica
```

Sin `--confirmar-evaluacion-unica` el script se niega a ejecutarse, y rechaza una carpeta de salida existente (`docs/aclaraciones/evaluacion_final_2025/`). `tests/test_evaluacion_final.py` cubre estas protecciones y la integridad del congelado sin cargar datos.
