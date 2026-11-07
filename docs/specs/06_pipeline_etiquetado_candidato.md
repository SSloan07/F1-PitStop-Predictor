# Especificación 06: pipeline de etiquetado candidato

**Identificador:** `SPEC-LABEL-001`  
**Versión:** 1.0  
**Estado:** contrato para implementar y revisar un etiquetado candidato; no declara etiquetas definitivas ni una matriz lista para entrenar.  
**Destinatario:** modelo o desarrollador que realizará la implementación.  
**Alcance de esta entrega:** inventario de eventos, contraste externo acotado, clasificación candidata, etiqueta por horizonte e informe reproducible.

## 1. Objetivo y referencias

Construir para cada piloto y vuelta una variable booleana nullable que indique si, al inicio de esa vuelta, se producirá una parada con cambio de neumáticos en esa vuelta o en alguna de las dos siguientes. El etiquetado puede consultar toda la información histórica; esa autorización no se extiende a las entradas futuras de un modelo predictivo.

Esta especificación operacionaliza [la propuesta de evidencia](03_etiquetado_candidato.md) y respeta [el problema](01_problema.md), [el pipeline de datos](02_pipeline_datos.md) y [el diseño experimental](05_diseno_ortogonal_experimentos.md). El [contraste inicial](../analisis/07_contraste_eventos_neumaticos.md) contiene casos de partida, no una validación general. Ante discrepancias, conservar la definición del horizonte de la especificación 01 y registrar la discrepancia antes de cambiar una regla.

Los valores de muestra y aceptación establecidos aquí son decisiones operativas para esta versión candidata. No son estimaciones de precisión ni umbrales científicos ya validados.

## 2. Alcance autorizado y límites

- **REQ-01:** implementar y ejecutar el pipeline en local; usar los archivos existentes. La consulta de fuentes públicas para revisar casos forma parte de esta tarea.
- **REQ-02:** desarrollar y revisar reglas únicamente con 2022–2023. Filtrar las filas antes de seleccionar casos o calcular resultados. No inspeccionar 2024–2025 para corregir o elegir reglas en esta entrega. La aplicación posterior de una regla congelada a esas temporadas será una ejecución identificable aparte.
- **REQ-03:** conservar una fila por temporada, carrera, piloto y vuelta. No crear una fila adicional con la versión completada ni generar vueltas ausentes.
- **REQ-04:** generar artefactos nuevos, con procedencia y versión. Conservar los archivos de entrada. Las etiquetas se publicarán como candidatas incluso si todos los controles de esta especificación pasan.
- **REQ-05:** esta tarea no incluye entrenamiento, selección de arquitecturas o métricas predictivas, ventanas de entrada, frontend, publicación en Drive ni push. La alineación temporal de predictores es una etapa posterior.

## 3. Entrada y configuración

Entrada principal, relativa a la raíz del repositorio:

`EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_limpieza_v1.pkl`

Entradas auxiliares de solo lectura: informe de limpieza, auditoría de reconstrucción de `Compound`, cachés locales de FastF1 y documentos de análisis. La falta de una entrada auxiliar se reporta; no se simula su contenido. Si falta la matriz principal, detenerse con un error que identifique la ruta.

Verificar al inicio claves completas y únicas, temporadas disponibles, tipos y orden temporal. Normalizar `DriverNumber` como identificador de texto sin mezclar pilotos por su código o equipo. `LapNumber` debe ser entero positivo, aunque esté almacenado como float; un valor fraccionario es un error de integridad. Un nulo de `PitInTime`/`PitOutTime` representa ausencia de registro de evento, no un tiempo cero.

Columnas necesarias: claves `Temporada`, `Ronda`, `DriverNumber`, `LapNumber`; tiempos `LapStartTime`, `Time`, `PitInTime`, `PitOutTime`; evidencia `Stint`, `Compound`, `TyreLife`, `FreshTyre`, `FastF1Generated`. Conservar `GranPremio`, `TrackStatus` e `IsAccurate` como contexto. Si falta una columna requerida, fallar antes de publicar resultados.

Los valores nulos de indicadores como `FastF1Generated` son desconocidos: no convertirlos silenciosamente en `False`. Documentar la conversión de tiempos a un tipo comparable de sesión; no comparar duraciones como texto ni confundirlas con la hora local de una crónica.

La configuración versionada debe incluir como mínimo:

```json
{
  "rule_version": "candidate_v1",
  "seasons": [2022, 2023],
  "horizon_laps": 3,
  "prediction_instant": "lap_start",
  "event_lap": "pit_entry",
  "seed": 42,
  "random_cases": 24,
  "directed_cases_per_stratum": 2,
  "max_sources_per_case": 3,
  "automatic_pattern": "entry_exit_immediate_stint_increase"
}
```

El horizonte es una definición del objetivo, no un hiperparámetro a optimizar en esta tarea. Registrar rutas de entrada/salida, versión de protocolo y huellas de archivos junto con esta configuración.

## 4. Unidad y convención temporal

Sea `K = (Temporada, Ronda, DriverNumber, N)` una muestra al inicio de la vuelta N. Su horizonte es el conjunto de números de vuelta:

`H(N) = {N, N+1, N+2}`.

No equivale a tomar las tres próximas filas: puede haber huecos, orden incorrecto o filas generadas. Las consultas y asociaciones siempre se limitan al mismo piloto y carrera.

- Una entrada en vuelta k se atribuye a k y puede activar etiquetas al inicio de k−2, k−1 y k, si esas muestras existen y son elegibles.
- La salida en k+1 aporta evidencia del mismo evento; no produce una segunda parada.
- Una salida inicial antes de cualquier entrada identificada no constituye por sí sola una parada de carrera.
- Una narración que mencione k+1 no desplaza automáticamente una entrada registrada en k. Contrastar la tabla de paradas y los tiempos cuando existan; si la correspondencia no se resuelve, el caso queda ambiguo.
- Un cambio durante una suspensión sin paso por boxes identificable no se convierte automáticamente en una parada objetivo. Los horizontes afectados por una atribución dudosa quedan desconocidos y se informan aparte.

## 5. Etapas del pipeline y responsabilidad única

Implementar funciones separadas para validar entradas, extraer eventos, seleccionar casos, leer evidencia revisada, clasificar eventos, calcular horizontes y emitir informes. El punto de entrada coordina esas funciones. Un notebook puede mostrar resultados, pero la lógica debe poder ejecutarse desde un script sin estado previo.

### 5.1 Inventario y asociación de eventos

**REQ-06:** crear un evento por cada registro de entrada y un inventario auxiliar de salidas huérfanas o transiciones de neumáticos sin entrada asociada. Usar identificadores estables derivados de la clave y el tipo de evento, no de la posición de una fila.

Ordenar entradas y salidas por su tiempo de sesión y emparejar cada entrada con la primera salida posterior del mismo piloto y carrera, antes de una nueva entrada. Una salida no puede pertenecer a dos entradas. No exigir que entrada y salida estén en filas distintas: pueden registrarse en la misma vuelta. Si faltan tiempos, hay órdenes imposibles, interrupciones no reconciliadas o más de una asociación posible, conservar la anomalía y marcarla desconocida. No unir automáticamente una entrada con una salida lejana tras otra entrada.

Registrar valores antes/después de stint, compuesto y edad, además de su procedencia. Describir aumentos de stint en las siguientes una y dos vueltas consecutivas para investigar retrasos. Un aumento distante no se atribuye automáticamente. Las transiciones sin paso por boxes, especialmente alrededor de bandera roja, requieren revisión.

El estado por vuelta de `TrackStatus` solo indica contexto en esa vuelta. Para afirmar el estado en un instante concreto hace falta la línea temporal de sesión/pista; si no está disponible, registrar el contexto como aproximado. Una fila generada o `IsAccurate=False` no prueba abandono ni ausencia de parada.

### 5.2 Muestra de revisión finita

**REQ-07:** guardar el inventario, la configuración y los identificadores seleccionados antes de buscar evidencia externa. Usar claves ordenadas y semilla 42. Un mismo evento no cuenta varias veces para alcanzar el tamaño de muestra.

Seleccionar 24 entradas al azar de 2022–2023 y hasta 16 casos dirigidos adicionales: dos por cada estrato siguiente, excluyendo los ya seleccionados:

1. mismo compuesto antes/después;
2. distinto compuesto;
3. posible juego usado (`FreshTyre=False` después);
4. transición de stint retrasada;
5. entrada sin transición inmediata, posible paso sin cambio o penalización;
6. entrada sin salida posterior o cerca del retiro/final;
7. paso o transición alrededor de una suspensión;
8. salida huérfana o transición de stint sin entrada.

Si un estrato no tiene casos, declararlo sin inventar ejemplos ni sustituirlo silenciosamente por otro estrato. Mantener como casos de regresión los ya documentados de Verstappen y Albon; el caso Latifi conserva su discrepancia hasta resolverla. Identificar cuáles se solapan con la muestra seleccionada.

Consultar hasta tres fuentes pertinentes por caso: priorizar informes de equipo, registros oficiales detallados y evidencia audiovisual identificable. Un listado de paradas solo verifica paso por boxes; una gráfica derivada de FastF1 no es confirmación independiente. Registrar dependencia conocida o incierta entre fuentes. Evitar transcripciones extensas de material protegido; guardar referencias y una síntesis propia.

Guardar por caso: URL y título, fecha de consulta, piloto/carrera, vuelta o tiempo citado, evidencia de cambio o de ausencia de cambio, convención temporal, decisión y limitaciones. Si no se puede acceder a una fuente o esta no decide el servicio realizado, el caso es `no_verificable`. No usar la salida de la propia regla como referencia externa ni sustituir un caso no verificable por otro sin dejar rastro.

### 5.3 Clasificación de eventos y habilitación de la regla

**REQ-08:** separar el estado del evento de la base que lo justifica:

| Campo | Valores permitidos |
| --- | --- |
| `event_status` | `change`, `no_change`, `unknown` |
| `evidence_basis` | `external_review`, `candidate_rule`, `unresolved` |
| `review_status` | `verified_change`, `verified_no_change`, `conflict`, `not_verifiable`, `not_reviewed` |

Una revisión externa coherente puede establecer `change` o `no_change`; una contradicción no resuelta establece `unknown`. La hipótesis automática R1 propone `change` cuando hay entrada/salida temporalmente emparejadas, aumento inmediato de stint en la vuelta consecutiva, filas no generadas y ninguna contradicción o interrupción pendiente de atribución. El mismo compuesto, un juego usado o una edad ausente no descartan R1. La falta de aumento inmediato, una transición retrasada, `TyresNotChanged` o una caída aislada de edad no permiten clasificar `no_change` automáticamente.

Para habilitar R1 como **regla candidata**, exigir todos estos controles fijados antes del contraste:

- al menos 10 eventos verificables de la muestra aleatoria donde R1 propone cambio;
- al menos dos cambios verificados al mismo compuesto y dos a juegos usados entre los casos revisados; pueden solaparse;
- al menos un paso por boxes verificado sin cambio de neumáticos, frente al cual R1 no proponga cambio;
- ningún cambio propuesto por R1 contradice un caso revisado verificable, incluidos los casos dirigidos;
- todas las atribuciones temporales utilizadas para evaluar R1 están reconciliadas; los casos dudosos quedan excluidos de esa evaluación y reportados;
- la regla R1 efectivamente revisada, sus excepciones y su versión coinciden con el código ejecutado.

Estos controles habilitan experimentación provisional, no certifican precisión poblacional. Informar denominadores, no verificables y discrepancias por separado para muestra aleatoria y dirigida. No presentar un porcentaje sobre casos elegidos por conveniencia como rendimiento general.

Si los controles fallan o falta evidencia, **terminar igualmente la entrega**: emitir inventario, revisión e informe y construir etiquetas con los eventos verificados externamente; el resto queda `unknown`. No habilitar R1, bajar el criterio ni buscar indefinidamente casos adicionales para conseguir la aprobación. Registrar `rule_status = requires_review`. Si pasan, registrar `rule_status = candidate_enabled`; los eventos inferidos por R1 llevarán siempre `evidence_basis = candidate_rule`. Una modificación de R1 requiere nueva versión y reevaluación sobre los mismos casos, con registro de las correcciones.

### 5.4 Cobertura y etiquetas por horizonte

**REQ-09:** calcular la etiqueta desde el inventario clasificado y un registro explícito de cobertura. Preservar los eventos ambiguos; nunca tratarlos como una lista vacía de paradas.

Para afirmar cobertura suficiente de un horizonte negativo, exigir sus tres números de vuelta presentes y no generados, intervalos temporales conocidos y coherentes (`LapStartTime` y `Time`), y ninguna interrupción, salida huérfana, transición de stint sin atribuir o evento sin resolver que pueda pertenecer al intervalo. Una entrada previa aún no cerrada también afecta la cobertura. Registrar las causas de insuficiencia; no deducir cobertura solo de que existan tres filas. La ausencia de compuesto o de edad por sí sola no invalida un horizonte sin anomalías de eventos.

La ausencia inferida bajo esas condiciones es una decisión según cobertura de las fuentes observadas, no una confirmación física independiente de cada vuelta. Registrar esta base como `source_coverage` y revisar la concordancia de las fuentes en el informe.

Aplicar este orden:

1. Si la muestra actual es generada, su instante inicial es inválido o no se puede determinar si corresponde a una predicción real: etiqueta `pd.NA`, motivo de exclusión.
2. Si existe al menos un evento `change` atribuido a N, N+1 o N+2: `True`. Esto prevalece sobre otra ambigüedad o un horizonte que se corta después del cambio observado. Registrar si depende de R1 o de evidencia externa.
3. Si no hay cambio, la cobertura es suficiente y todos los eventos del intervalo son `no_change`: `False`. También aplica cuando no hubo entradas y la cobertura es suficiente.
4. En cualquier otro caso: `pd.NA` con motivo.

**REQ-10:** al final de carrera o tras el último registro del piloto, no inventar las vueltas faltantes ni etiquetar `False` por defecto. Un positivo observado dentro del horizonte sigue siendo válido; un horizonte incompleto sin positivo queda desconocido en v1. No declarar abandono mediante un umbral de tiempo improvisado.

## 6. Contratos de salida

Generar un directorio de ejecución que identifique configuración, versión y huella de entrada. Si ya existe, verificar identidad y reutilizarlo o rechazar la escritura; no sobrescribir silenciosamente resultados diferentes. El pipeline offline debe poder repetirse usando el registro de evidencia, sin volver a consultar la web.

| Artefacto | Contenido obligatorio |
| --- | --- |
| `events.csv` | `event_id`, claves de carrera/piloto, vuelta/tiempo de entrada y salida, señales antes/después, `event_status`, `evidence_basis`, `review_status`, `reason`, versión de regla y referencias de evidencia. |
| `review_cases.json` | selección congelada, estratos, fuentes y decisión de cada caso, acceso fallido, dependencia de fuente y discrepancias. |
| `coverage.csv` | claves piloto-vuelta, estado de cobertura y motivos; referencias a anomalías/eventos relevantes. |
| `labels.pkl` | claves originales más `PitWithin3Laps` de dtype pandas `boolean` nullable, `LabelReason`, `LabelBasis`, `RuleVersion` y `EligibleForSupervisedLearning`. Una fila por muestra original de las temporadas seleccionadas. |
| `manifest.json` | rutas y SHA-256 de entradas/salidas, configuración, versión del código/protocolo, entorno/dependencias, temporadas incluidas, filtros, fechas y estado de habilitación de la regla. |
| `report.md` | integridad, tamaño de muestra real, controles de R1, desacuerdos, recuentos de eventos/etiquetas/motivos por temporada, carrera y piloto, ejemplos y limitaciones. |

`LabelBasis` debe distinguir `external_review`, `candidate_rule`, `source_coverage` y `unresolved`. Para positivos con varias evidencias, conservar referencias completas en auditoría y declarar si alguna externa basta para sostenerlos. `EligibleForSupervisedLearning` es verdadero solo para etiquetas conocidas en muestras actuales válidas; no constituye certificación de que las variables ya sean utilizables en entrenamiento.

Usar códigos estables en `LabelReason`, como `change_in_horizon`, `complete_horizon_no_change`, `ambiguous_event`, `incomplete_horizon`, `invalid_current_sample` o `insufficient_coverage`. Si concurren varios motivos, conservar todos en auditoría y aplicar la precedencia de la sección 5.4 para el motivo principal. Los CSV/JSON deben expresar los desconocidos como nulos o estados explícitos, nunca como el texto booleano `False`.

No añadir las etiquetas mediante un join que multiplique filas. Las filas sin etiqueta permanecen en el artefacto de auditoría. El archivo `labels.pkl` contiene claves y etiquetas, no una exportación automática de predictores. Cualquier unión posterior con la matriz debe validar relación uno a uno.

## 7. Prevención de fuga de información

**REQ-11:** etiquetar puede usar futuro histórico. El resultado no autoriza utilizar como predictores `PitInTime`, `PitOutTime`, el estado del evento, motivos, evidencias, cobertura del horizonte, resultado final del piloto ni columnas derivadas del objetivo. La etiqueta y sus metadatos deben permanecer fuera de cualquier selección de entradas de un modelo.

`Stint`, `Compound`, `TyreLife` y clima solo podrán ser predictores si su valor estaba disponible al inicio de la vuelta correspondiente; una reconstrucción retrospectiva no acredita esa disponibilidad. Los resúmenes de una vuelta completa solo están disponibles después de terminarla. Esta entrega identifica el límite y no declara resuelto el alineamiento temporal de esas variables.

## 8. Verificaciones de aceptación

**REQ-12:** implementar pruebas pequeñas y deterministas con eventos sintéticos; el acceso a web y API no debe ser necesario para ejecutarlas. Cubrir al menos:

| Escenario | Resultado esperado |
| --- | --- |
| Cambio en k, muestras k−2, k−1 y k válidas | Tres positivos; k+1 no recibe ese evento. |
| Entrada en k y salida en k+1 | Un evento, atribuido a k. |
| Entrada y salida en la misma vuelta con orden temporal válido | Un evento; no perderlo por comparar solo filas vecinas. |
| Salida inicial previa a cualquier entrada | No crear parada por ese registro aislado. |
| Mismo compuesto y evidencia de cambio | No descartar el cambio por igualdad de compuesto. |
| `FreshTyre=False` con cambio verificado | No descartar el cambio por juego usado. |
| Stint sin aumento o con aumento retrasado sin revisión | Evento desconocido; no inferir ausencia de cambio. |
| Evento ambiguo dentro de horizonte sin positivo | Etiqueta ausente. |
| Positivo y evento ambiguo dentro del mismo horizonte | Positivo en muestra actual válida. |
| Hueco o final incompleto sin positivo | Ausente; no usar las siguientes tres filas ni rellenar `False`. |
| Muestra actual generada o sin instante válido | Ausente y no elegible. |
| Dos entradas consecutivas, una salida o tiempos inválidos | Anomalía, sin reutilizar salida ni unir a través de otra entrada. |
| Aumento de stint sin boxes durante suspensión | No crear positivo automático. |
| Datos de otro piloto/carrera o entrada desordenada | Ninguna asociación cruzada; resultado estable al ordenar. |
| Control de R1 no satisfecho | Sin clasificaciones automáticas; solo decisiones externas verificadas. |

Además, verificar en los datos reales: claves y número de filas del subconjunto preservados, exclusión efectiva de 2024–2025 de la revisión, dtype nullable, ningún desconocido convertido en falso y correspondencia entre los recuentos de reporte y artefactos. El manifiesto y las fuentes deben permitir rastrear cada etiqueta positiva hasta su evento. Repetir con las mismas entradas debe producir las mismas decisiones e identificadores; las fechas de ejecución pueden variar.

## 9. Organización y ejecución esperada

Crear módulos según responsabilidades, por ejemplo en `src/f1_pitstop/labeling/`, una configuración en `configs/` y un script delgado `scripts/etiquetar.py`. Añadir solo el soporte de importación y dependencias necesario; no reestructurar todo el repositorio como parte de esta entrega. Los tests sintéticos se ubicarán en `tests/labeling/`.

Documentar un comando para generar inventario/muestra, cómo completar el registro de evidencia y otro comando offline para clasificar y producir etiquetas e informe. Las rutas y opciones deben ser explícitas; comprobar los comandos implementados. No presentar los nombres propuestos aquí como herramientas ya existentes.

Al terminar: explicar qué se implementó, qué controles pasaron, si R1 quedó habilitada y cuántos casos continúan desconocidos. Registrar cambios importantes en commits breves en inglés, sin prefijos como `feat:`; incluir solo código/documentación pertinente. Los datos grandes y cachés siguen como artefactos locales. No hacer push.

## 10. Encargo para el modelo ejecutor

> Implementa y ejecuta `SPEC-LABEL-001` en este repositorio. Lee primero esta especificación, las referencias de la sección 1 y las instrucciones locales aplicables. Entrega inventario, revisión externa acotada a 2022–2023, regla candidata evaluada, etiquetas nullable, manifiesto e informe, junto con verificaciones reproducibles. Trata las decisiones no verificables como desconocidas y conserva las discrepancias. Si R1 no supera sus controles, completa la entrega con evidencia externa verificada y deja los demás casos sin clasificar; explica el límite. No entrenes modelos, no ajustes reglas con 2024–2025 y no publiques etiquetas como definitivas. Sigue las responsabilidades y contratos de esta especificación y registra el trabajo en commits sin push.
