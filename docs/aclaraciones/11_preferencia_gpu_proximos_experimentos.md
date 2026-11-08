# Entorno de los próximos experimentos: GPU

**Decisión vigente del usuario:** utilizar GPU explícitamente para los próximos experimentos. Esta aclaración sustituye la interpretación anterior de usar CPU.

- Las configuraciones nuevas usarán `training.device="cuda"`, en lugar de depender de `auto`.
- Si CUDA no está disponible, el entrenamiento debe detenerse antes de comenzar; no se sustituirá GPU por CPU automáticamente.
- Las referencias y candidatas que se comparen deberán ejecutarse en GPU y compartir modelo de GPU, versiones y controles de entorno.
- El dispositivo efectivo se comprobará en `environment.json` de cada corrida y en los registros de entorno de las parejas.
- Un runtime de Colab con GPU asignada o un indicador de memoria GPU no prueba que el entrenamiento de PyTorch la haya utilizado.

Esta decisión no reescribe el historial: la primera corrida F2 registró T4/CUDA y la corrida F1 registró CPU. La pareja RNN32–RNN16 distribuida declara `auto`; sus artefactos confirman GPU Tesla T4/CUDA en ambos hijos. Ver [evaluación pareada](12_resultados_rnn16_pareada.md). No se modificaron sus configuraciones ni su ZIP después de la publicación.
