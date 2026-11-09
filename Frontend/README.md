# Pitwall · replay 2024

Reproducción histórica por tiempo de sesión sobre la validación 2024. El selector usa salidas de inferencia exportadas de modelos distintos; no entrena ni recalcula checkpoints. Cada predicción se revela a partir del `decision_time` original de la muestra. Las etiquetas futuras no se sirven al navegador; la parada real solo aparece al llegar a su `entry_time`.

## Ejecutar

Desde la raíz del repositorio, con Python 3.13 y pandas instalados:

```bash
python3.13 -m pip install -r requirements-simulator.txt
python3.13 scripts/servir_simulador_2024.py
```

Abrir `http://127.0.0.1:8765`. El backend valida SHA-256 de la matriz 2024, eventos, índice causal y ZIPs de modelo al iniciar. Los ZIPs de `runs/`, la matriz normalizada y `EDA/dataset/cache_fastf1_carreras/2024/` son artefactos locales que no se suben a Git. La vista de pista necesita las cachés `position_data.ff1pkl` y `car_data.ff1pkl` para las 24 carreras 2024.

## Alcance

- Ocho checkpoints disponibles: RNN y LSTM, incluidos objetivos F1/F2 y variantes de regularización/pérdida.
- Todos exponen predicciones de las mismas 25.635 muestras de validación y 24 carreras de 2024.
- El mapa anima los autos con coordenadas X/Y y velocidad de las cachés locales FastF1 de 2024; el trazado se deriva de una vuelta registrada. Halo rojo = alerta, ámbar = entrada real al pit lane y verde = detención observada a ≤2 km/h dentro del intervalo PitIn/PitOut. La posición de la tabla sigue siendo la última vuelta cerrada.
- La alerta es riesgo de entrada a boxes en la vuelta de decisión o en las dos siguientes; puede haber varios pilotos sobre el umbral.
- Métricas y cobertura de 2024 son desarrollo, no una prueba independiente. 2025 no participa en la interfaz, el replay ni sus métricas.
- Es un replay de resultados históricos exportados, no una inferencia de feed en vivo.


Las posiciones del mapa se actualizan únicamente con muestras ya observadas según el reloj del replay. Los cambios visuales entre muestras se suavizan en pantalla; esto no cambia el tiempo de decisión ni las alertas del modelo. El inicio de pit lane se revela en `entry_time`; la detención física se marca cuando la telemetría registra velocidad ≤2 km/h.
