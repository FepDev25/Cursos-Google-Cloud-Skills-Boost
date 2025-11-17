# ¿Qué problema está resolviendo `tf.data`?

- Cuando trabajas con datasets grandes, NO puedes simplemente hacer:

```python
x = pd.read_csv("100GB_file.csv")
```

- Porque:
  - No cabe en memoria
  - No puedes entrenar cargando todo de golpe
  - Entrenar sería lento

- TensorFlow soluciona esto con `tf.data.Dataset`:
  - es una abstracción que representa un flujo de datos, no una carga completa.

## ¿Qué es un `Dataset`?

- Un `Dataset` es una secuencia de elementos, donde cada elemento puede tener uno o varios tensores.

- Ejemplos:

| Tipo de proyecto          | Elemento del dataset        |
| ------------------------- | --------------------------- |
| Clasificación de imágenes | `(imagen_tensor, etiqueta)` |
| NLP                       | `(texto, etiqueta)`         |
| Tabular                   | `(features_vector, label)`  |

## Formas de crear un Dataset

- Hay dos formas principales:

| Tipo                    | Qué significa                                          | Ejemplo                                                      |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| Data source         | Creas el dataset desde cero leyendo archivos o memoria | `TextLineDataset`, `TFRecordDataset`, `from_tensor_slices()` |
| Data transformation | Partes de un dataset existente y le aplicas etapas     | `map()`, `batch()`, `shuffle()`                              |

### Ejemplo simple usando memoria

```python
data = tf.data.Dataset.from_tensor_slices([1,2,3,4])
```

## Datasets grandes → “Sharded”

- Cuando tienes datos enormes, no están en un único archivo, sino en muchos, por ejemplo:

```bash
images-0001.tfrecord
images-0002.tfrecord
images-0003.tfrecord
...
```

- Esto permite:
  - Cargar progresivamente
  - Distribuir en múltiples máquinas
  - Paralelizar lectura

- Solo necesitas un batch para entrenar un paso, no todo el dataset en memoria.

## Pipeline típico

- Un pipeline real con `tf.data` se ve así:

```bash
Leer archivos → mezclar → procesar → batch → alimentar modelo
```

- Ejemplo:

```python
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.shuffle(1000)
dataset = dataset.map(parse_example)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

## Clases importantes de lectura

| Dataset class                  | Para                                                |
| ------------------------------ | --------------------------------------------------- |
| `TextLineDataset`              | Leer texto línea por línea (CSV, texto plano)       |
| `TFRecordDataset`              | Leer formato binario optimizado de TensorFlow       |
| `FixedLengthRecordDataset`     | Leer registros de tamaño fijo (audio, imágenes RAW) |
| `Dataset.from_tensor_slices()` | Dataset desde memoria (para datos pequeños)         |

## ¿Cómo funciona internamente?

- A nivel conceptual:

1. Creas dataset (`TFRecordDataset`)
2. Lo transformas (`shuffle`, `map`, `batch`)
3. TensorFlow crea un iterador
4. El modelo pide un batch → TensorFlow ejecuta `next()` internamente
5. TensorFlow alimenta el batch al entrenamiento
6. Al terminar, libera memoria del iterador

- Esto permite streaming constante de datos sin saturar RAM.

## Importante: Iteradores

- TensorFlow usa iteradores especiales, optimizados para:
  - Memoria
  - Paralelismo
  - GPU streaming (prefetch)

- Cuando ya no se usan → se destruyen para liberar memoria.

## Características clave de `tf.data`

| Función      | Beneficio                                                            |
| ------------ | -------------------------------------------------------------------- |
| `shuffle()`  | Evitar patrones en los datos                                         |
| `batch()`    | Entrenar con múltiples ejemplos por paso                             |
| `map()`      | Aplicar transformaciones (augmentación, tokenización, normalización) |
| `prefetch()` | Preparar siguiente batch mientras modelo entrena                     |
| `cache()`    | Guardar transformaciones para acelerar                               |

## Resumen

| Concepto                                         | Respuesta                                                              |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| ¿Qué es `tf.data.Dataset`?                       | Una representación de una secuencia de elementos (ej: `(x,y)` pares)   |
| ¿Por qué no cargamos todo el dataset en memoria? | Porque puede ser demasiado grande y solo necesitamos un batch por paso |
| ¿Qué formatos soporta?                           | Texto, CSV, TFRecord, binarios                                         |
| ¿Qué método alimenta el modelo?                  | `model.fit(dataset)`                                                   |
| ¿Qué transforma datos?                           | `map()`                                                                |
| ¿Qué agrupa?                                     | `batch()`                                                              |
| ¿Qué mejora velocidad?                           | `prefetch()` y `num_parallel_calls`                                    |
