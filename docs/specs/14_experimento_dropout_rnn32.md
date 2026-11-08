# Experimento de regularización: dropout en RNN32

**Estado:** preparado para entrenamiento en Colab; no contiene resultados. 2025 permanece reservada.

## Pregunta

¿Agregar dropout de 0,2 a la capa de fusión mejora la generalización y el F2 de validación de la referencia RNN32? La comparación usa dos entrenamientos nuevos en la misma sesión GPU: referencia sin dropout (0,0) y candidata con dropout (0,2). La candidata conserva la arquitectura y dimensiones; dropout se aplica después de ReLU en la capa de fusión. No se aplica dropout recurrente.

## Controles y factor

Se mantienen las características, máscaras, ventanas W=8, preprocesador congelado, particiones por carrera, población elegible, semilla 42, Adam, learning rate 0,001, BCE sin ponderación, batch 256, recorte de gradiente 1, máximo 30 épocas, paciencia 6 y reglas de selección. La pérdida de entrenamiento continúa siendo BCE. Early stopping selecciona checkpoint por F2 positiva a umbral fijo 0,5 y desempata por menor BCE de validación. Congelados los pesos, cada hijo selecciona el umbral F2 con la cuadrícula vigente.

| Factor | Referencia | Candidata |
|---|---:|---:|
| Unidades recurrentes | 32 | 32 |
| Dropout después de ReLU en la fusión | 0,0 | 0,2 |
| Seed | 42 | 42 |
| Entrenamiento | GPU CUDA obligatoria | Misma GPU y sesión |

El ejecutor rechaza configuraciones con otras diferencias materiales y falla antes de entrenar si CUDA no está disponible. Comprueba versiones, dispositivo, GPU, cohorte, objetivos, manifiestos, fuentes y configuraciones efectivas antes de autorizar el bundle pareado. Las máscaras de dropout son el tratamiento experimental; el mismo seed controla inicialización y orden de lotes, pero no hace iguales las trayectorias estocásticas.

## Datos y evaluación

Train comprende carreras 2022–2023 y validación carreras 2024. 2025 no se empaqueta, transforma, selecciona ni evalúa. El ajuste del preprocesador ya está congelado y se hizo únicamente con train. Se reportan F2 como métrica primaria, F1, precision, recall, AP, BCE, alertas, matrices de confusión y cobertura por evento en la evaluación local. Validación participa en early stopping y selección de umbral, así que sus métricas son de desarrollo, no una estimación final independiente.

## Ejecución

Desde la raíz:

```bash
python scripts/empaquetar_experimento_dropout.py \
  --output runs/rnn_colab_dropout_v2.zip
```

Registrar el SHA-256 impreso, subir el ZIP a Colab, seleccionar GPU y ejecutar [`04_rnn_dropout_colab.ipynb`](../../notebooks/04_rnn_dropout_colab.ipynb). El notebook comprueba el manifiesto y hashes, valida `nvidia-smi` y conserva el PyTorch CUDA preinstalado por Colab: solo instala NumPy y pandas. Se detiene si CUDA no está disponible. La salida esperada es `rnn32_dropout02_f2_v1_seed42.zip`; devolverla sin modificar al directorio local `runs/` para evaluar ambos hijos en CPU con el pipeline existente.

```bash
python scripts/entrenar_comparacion_dropout.py \
  --reference-config configs/experimento_dropout_ref_v1.json \
  --candidate-config configs/experimento_dropout_02_v1.json \
  --output runs/rnn32_dropout02_f2_v1_seed42
```

La evaluación local usa el mismo holdout 2024 y umbrales congelados. El bundle de Colab incluye pesos y predicciones de validación, pero ningún array de test. El ZIP padre registra checksums, controles, entorno y estado de autorización.

## Interpretación y límites

La comparación estima el efecto de aplicar dropout a la cabeza de fusión bajo esta configuración y semilla. Un cambio en F2 de una sola semilla no demuestra estabilidad. AP/BCE complementan el criterio principal, pero no lo reemplazan. No se cambiarán arquitectura, weight decay, pérdida, features, semilla ni selección dentro de esta corrida. La ejecución en GPU y la evaluación local no hacen de 2024 un conjunto final ciego. El siguiente cambio arquitectónico (por ejemplo, LSTM) debe quedar como otro nodo del árbol y mantener explícito este resultado, gane o no.

Referencias internas: [F2 inicial](../aclaraciones/11_experimento_rnn_inicial.md) y [contrato de preprocesamiento](10_contrato_preprocesamiento.md).
