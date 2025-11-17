# Working in-memory and with files

## Datasets cargados desde memoria

- Si los datos ya están en memoria, TensorFlow ofrece dos métodos:

| Método                                 | Qué hace                                                               | Ejemplo                                  |
| -------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------- |
| `tf.data.Dataset.from_tensors()`       | Crea un dataset con un único elemento, que contiene todo el tensor | Todo el dataset como una sola unidad |
| `tf.data.Dataset.from_tensor_slices()` | Crea un dataset donde cada fila es un elemento                     | Lo más usado para modelos                |

- Ejemplo:

```python
data = tf.constant([[1,2],[3,4],[5,6]])

ds1 = tf.data.Dataset.from_tensors(data)
ds2 = tf.data.Dataset.from_tensor_slices(data)
```

| Propiedad                | `from_tensors` | `from_tensor_slices` |
| ------------------------ | -------------- | -------------------- |
| Número de elementos      | 1              | 3                    |
| Usado para entrenamiento | raro         | común              |

## Trabajar con archivos → TextLineDataset

- Cuando los datos están en archivos (como CSV), usamos:

```python
dataset = tf.data.TextLineDataset("data.csv")
```

- Esto lee línea por línea, no todo el archivo de golpe → perfecto para datasets grandes.
- Después se convierte la línea en datos útiles usando `map()`:

```python
dataset = dataset.map(parse_csv)  # transforma texto → tensores
```

## Flujo típico para un dataset desde archivo

```bash
Text file → TextLineDataset → map(parse) → shuffle → batch → prefetch → model.fit()
```

Ejemplo real:

```python
def parse_csv(line):
    parsed = tf.io.decode_csv(line, record_defaults=[0., 0., 0.])
    features = parsed[:-1]
    label = parsed[-1]
    return features, label

dataset = tf.data.TextLineDataset("train.csv")

dataset = dataset.map(parse_csv)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

## Importante: Solo se hace `shuffle()` al dataset de entrenamiento

Porque si lo haces también a validación o test → invalidas su uso para medir rendimiento real.

Esto puede verse como:

```python
if training:
    dataset = dataset.shuffle(1000)
```

## Trabajar con datasets sharded (divididos en muchos archivos)

- Cuando hay muchos archivos:

```bash
data-0001.csv
data-0002.csv
data-0003.csv
...
```

- Primero se listan:

```python
files = tf.data.Dataset.list_files("data-*.csv")
```

- Luego cada archivo se convierte a dataset usando `TextLineDataset`.

- Pero aquí ocurre algo importante:

| Transformación | Cuándo usarla                                 |
| -------------- | --------------------------------------------- |
| `map()`        | cuando una entrada produce una salida     |
| `flat_map()`   | cuando una entrada produce muchas salidas |

- Ejemplo:

- 1 archivo → muchas líneas → flat_map

```python
dataset = files.flat_map(tf.data.TextLineDataset)
```

- Luego parseamos las líneas:

```python
dataset = dataset.map(parse_csv)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

## Prefetching → eficiencia

- Con CPU y GPU:

| Sin prefetch                                    | Con prefetch                       |
| ----------------------------------------------- | ---------------------------------- |
| GPU espera a CPU que procese el siguiente batch | GPU y CPU trabajan en paralelo |
| Entrenamiento lento                             | Entrenamiento optimizado           |

- Ejemplo recomendado siempre:

```python
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

- Esto hace que TensorFlow produzca el siguiente batch mientras el modelo entrena.

## Resumen

| Concepto               | Explicación                           | Ejemplo           |
| ---------------------- | ------------------------------------- | ----------------- |
| `from_tensor_slices()` | Dataset en memoria dividido por filas | Mini datasets     |
| `TextLineDataset()`    | Leer archivos línea por línea         | CSV, texto        |
| `list_files()`         | Leer múltiples archivos sharded       | `"*.csv"`         |
| `map()`                | Una entrada → una salida              | parsear CSV       |
| `flat_map()`           | Una entrada → muchas salidas          | archivo → dataset |
| Shuffle                | Solo para train                       | evita patrón      |
| Batch                  | Agrupa muestras                       | eficiencia        |
| Prefetch               | CPU/GPU paralelos                     | velocidad         |

## Concepto clave

- El objetivo de `tf.data` es mantener el modelo ocupado sin esperas, automatizando carga, batching, shuffling y preprocesamiento de datos en paralelo.
