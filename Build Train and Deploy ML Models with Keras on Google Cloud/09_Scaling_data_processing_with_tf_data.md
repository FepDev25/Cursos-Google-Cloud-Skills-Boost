# Scaling data processing with `tf.data` y capas de preprocesamiento de Keras

- Los datos son una parte fundamental en todo modelo de Machine Learning. No basta con obtener datos: hay que limpiarlos, transformarlos y prepararlos correctamente para que el modelo pueda aprender patrones útiles.
- En esta lección se habla de cómo escalar (hacer eficiente y automatizar) el preprocesamiento de datos usando:
- `tf.data` → para crear pipelines eficientes de entrada de datos.
- Keras Preprocessing Layers → capas que transforman los datos dentro del propio modelo.
- Cuando combinas estas herramientas con TensorFlow, puedes crear modelos de extremo a extremo, es decir:
  - Modelos que reciben datos crudos (imágenes, texto o numéricos)
  - Y los preprocesan internamente sin pasos manuales externos.

- Esto hace que el modelo:
  - sea más portable,
  - tenga menos errores al exportarse,
  - y evite diferencias entre entrenamiento y predicción ("training-serving skew").

## Tipos de capas de preprocesamiento disponibles

- Estas capas permiten transformar diferentes tipos de datos:

| Tipo de dato    | Capas comunes                                                  |
| --------------- | -------------------------------------------------------------- |
| Texto           | `TextVectorization`                                            |
| Datos numéricos | `Normalization`, `Discretization`                              |
| Categóricos     | `StringLookup`, `IntegerLookup`, `CategoryEncoding`, `Hashing` |
| Imágenes        | Capas de aumento y procesamiento (flip, resize, etc.)          |

### 1. Preprocesamiento de texto

- La capa TextVectorization` convierte texto crudo en una representación numérica que el modelo pueda usar.
- Puede generar:
  - una lista de índices enteros (tokens), o
  - una representación densa con valores flotantes.
- Ejemplo:
  - Texto → tokenización → números → capa *Embedding* o *Dense*.
- Opcionalmente, esta capa tiene un método llamado adapt()`, que analiza un dataset, calcula la frecuencia de palabras y crea un vocabulario.
- Puedes limitar el tamaño del vocabulario si quieres.

### 2. Preprocesamiento de datos numéricos

- La capa Normalization` estandariza valores numéricos para que:
  - Tengan media (µ) = 0
  - Desviación estándar (σ) = 1
- Esto mejora el entrenamiento y estabilidad del modelo.
- Funcionamiento:

```bash
valor_normalizado = (input - media) / sqrt(varianza)
```

- La capa también usa adapt()` para calcular la media y varianza antes del entrenamiento.

### 3. Discretización (buckets)

- La capa Discretization` convierte valores continuos en categorías por rangos (buckets).
- Ejemplo: edad
  - bucket 0: 0-10
  - bucket 1: 11-20
  - bucket 2: 21-30 …
- Esto es útil para variables numéricas sin escala clara.

### 4. Preprocesamiento de datos categóricos

- Capas disponibles:

| Capa               | Función                                            |
| ------------------ | -------------------------------------------------- |
| `CategoryEncoding` | Convierte categorías a one-hot, multi-hot o conteo |
| `Hashing`          | Convierte categorías usando hashing trick          |
| `StringLookup`     | Convierte strings a índices enteros                |
| `IntegerLookup`    | Convierte valores enteros categóricos en índices   |

- Estas también pueden usar `adapt()` para aprender un vocabulario.

## Capas con estado (`stateful`)

- Algunas capas almacenan información aprendida del dataset, como:
  - vocabulario de palabras (`TextVectorization`)
  - mapeo de categorías (`StringLookup`, `IntegerLookup`)
  - media y desviación estándar (`Normalization`)
  - límites de buckets (`Discretization`)
- Estas capas no se entrenan con el modelo, sino que deben configurarse antes usando:
  - inicialización manual
  - `adapt(dataset)`

## Ejemplo práctico (dataset PetFinder)

- Se usa un dataset donde cada fila representa una mascota, con atributos numéricos, texto y categorías.
- El objetivo es predecir si será adoptada.

- Ejemplo:
  - `"Dog"` o `"Cat"` → convertidos con `StringLookup` → one-hot encoding
  - números → normalizados con `Normalization`

## Dónde aplicar las transformaciones

- Hay dos formas principales:

### Opción 1: dentro del modelo

- Procesamiento en GPU
- Modelo portable y auto-contenido
- Mejor para imágenes y normalización

### Opción 2: usando `dataset.map()` en la tubería `tf.data`

- Procesamiento en CPU de forma asíncrona
- Puede mejorar velocidad usando `AUTOTUNE`
- Esta opción es ideal cuando procesas muchos datos paralelamente.

## Exportar modelos end-to-end

- Cuando el preprocesamiento está dentro del modelo, puedes exportarlo a:
  - TensorFlow Serving
  - TensorFlow Lite
  - TensorFlow.js
- Sin tener que re-escribir el código de preprocesamiento en otro lenguaje (como JavaScript).
- Esto reduce errores y asegura que las predicciones usen exactamente las mismas transformaciones usadas en entrenamiento.

## Resumen final

| Concepto                          | Idea clave                                                             |
| --------------------------------- | ---------------------------------------------------------------------- |
| Preprocesamiento con Keras Layers | Permite construir modelos que procesan datos crudos                    |
| `adapt()`                         | Aprende información estadística o vocabularios antes del entrenamiento |
| Opciones de procesamiento         | Dentro del modelo (GPU) o con `tf.data` (CPU)                          |
| Ventaja                           | Modelos portables, sin necesidad de repetir lógica al exportarlos      |
