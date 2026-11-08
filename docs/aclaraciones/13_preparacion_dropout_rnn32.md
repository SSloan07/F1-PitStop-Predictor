# Preparación de la comparación RNN32 con dropout

**Estado:** listo para Colab; todavía no se ha entrenado ni evaluado esta variante.

Se implementó una comparación pareada que reutiliza el loader, el entrenamiento, la política F2 y el formato de artefactos del proyecto. Ambos modelos se entrenan desde cero en la misma sesión. El único cambio es `model.dropout_rate`, de 0,0 a 0,2, aplicado después de ReLU en la capa de fusión. La RNN, sus 32 unidades, la rama estática, las features y la cabeza mantienen sus dimensiones.

Early stopping ya pertenecía al protocolo: máximo 30 épocas, paciencia 6 y checkpoint por F2 de validación a 0,5, con BCE menor para desempatar. La búsqueda de umbral ocurre después de congelar el checkpoint, sobre la cuadrícula F2 vigente. No se cambió esta regla.

El runner requiere `training.device=cuda` en las dos configuraciones y verifica que la GPU y las versiones no cambien entre modelos. No entrena localmente ni usa 2025. El empaquetador incluye únicamente datos congelados de train y validación, manifiesto, preprocesador y código necesario.

## Comando local

```bash
python scripts/empaquetar_experimento_dropout.py \
  --output runs/rnn_colab_dropout_v2.zip
```

Guardar el SHA-256 impreso. En Colab, seleccionar GPU, subir `runs/rnn_colab_dropout_v2.zip` y abrir [`04_rnn_dropout_colab.ipynb`](../../notebooks/04_rnn_dropout_colab.ipynb). El notebook verifica el paquete y consulta `torch.cuda.is_available()`, instala solo NumPy y pandas y conserva el PyTorch de Colab y detiene la ejecución si CUDA no está disponible.

Tras la corrida, descargar `rnn32_dropout02_f2_v1_seed42.zip` y colocarlo en `runs/`. La siguiente fase es evaluar ambos hijos localmente en CPU, verificar paridad, generar el informe y actualizar el árbol. Aún no se debe afirmar que dropout mejora el modelo.
