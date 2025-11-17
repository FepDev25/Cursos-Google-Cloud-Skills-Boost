# Getting the data ready for model training

## ¿Qué significa "preparar datos para entrenar"?

- Antes del modelo viene la pregunta clave:

> ¿Qué información (features) necesita el modelo para aprender?

- Ejemplo: predecir precio de propiedades.

Posibles features:

| Feature        | Tipo       | Ejemplo                   | ¿Puede usarse directamente? |
| -------------- | ---------- | ------------------------- | --------------------------- |
| Tamaño (m²)    | Numérico   | `134`                     | Sí                        |
| Tipo propiedad | Categórico | `"house"` o `"apartment"` | No directamente           |

- Los modelos no entienden texto → todo debe convertirse en números.

## Representar los features usando Feature Columns

- TensorFlow tiene un API llamada feature_columns que ayuda a transformar los datos en una forma que un modelo pueda usar.

## Numéricos

- Se usan tal cual:

```python
square_footage = tf.feature_column.numeric_column("sqft")
```

- Esto irá directo como un valor dentro del vector de entrada.

## Categóricos → deben transformarse

- Ejemplo: `"house"` vs `"apartment"`
- Necesitan convertirse a números antes de entrar al modelo.
- Hay varias formas según el caso:

| Tipo feature column                       | Cuándo usarlo                                                | Ejemplo                       |
| ----------------------------------------- | ------------------------------------------------------------ | ----------------------------- |
| `categorical_column_with_vocabulary_list` | Cuando tienes categorías conocidas almacenadas en código | `"house", "apartment"`        |
| `categorical_column_with_vocabulary_file` | Cuando la lista de categorías está en un archivo externo     | CSV, diccionario externo      |
| `categorical_column_with_identity`        | Cuando las categorías ya son números (`0..N`)                | IDs, clases numeradas         |
| `categorical_column_with_hash_bucket`     | Cuando las categorías son muy muchas o desconocidas (NLP)    | palabras, nombres de ciudades |

- Ejemplo:

```python
property_type = tf.feature_column.categorical_column_with_vocabulary_list(
    "type", ["house", "apartment"]
)
```

### ¿Cómo se representan dentro del modelo?

- Las columnas categóricas se convierten en:
  - One-hot vectors

- Ejemplo con 2 clases:

| Categoría   | One-hot |
| ----------- | ------- |
| `house`     | `[1,0]` |
| `apartment` | `[0,1]` |

- Luego el modelo puede multiplicar esos valores por pesos.

## Cómo usa esto un modelo (ej: Linear Regressor)

- Un modelo lineal recibe un vector numérico.
- Ejemplo final después del feature engineering:

```bash
[ sqft , type_house , type_apartment ]
```

- Ejemplo: 120m² y tipo "apartment":

```bash
[120, 0, 1]
```

- El modelo hace:

```bash
price = w1*120 + w2*0 + w3*1 + bias
```

- Después durante entrenamiento, los pesos se ajustan al problema.

## Feature Columns avanzadas

TensorFlow incluye más columnas útiles:

| Tipo                  | Para qué sirve                                              |
| --------------------- | ----------------------------------------------------------- |
| `bucketized_column()` | Convertir números continuos en rangos (ej: latitud, edades) |
| `embedding_column()`  | Reemplazar one-hot muy grandes por representaciones densas  |
| `crossed_column()`    | Combinar categorías (ej: ciudad + tipo propiedad)           |

## Ejemplo: bucketized (discretización)

En propiedades:

- Latitud = datos demasiado detallados
- Convierte en grupos → como códigos postales aproximados

```python
bucketized = tf.feature_column.bucketized_column(latitude, boundaries=[10,20,30])
```

Esto agrupa valores en rangos.

## Embeddings (fundamental en NLP y grandes categorías)

Si tienes:

```bash
1 millón de categorías
```

- one-hot sería:

```bash
[0, 0, 0, ..., 0, 1]
```

- Muy grande → ineficiente.

- Solución: embeddings
  - Ejemplo: cada categoría se representa como un vector pequeño como:

```bash
[0.32, -0.52, 1.8]
```

- Mucho más eficiente, y la red aprende relaciones entre categorías.

## Resumen

```bash
Raw Inputs
──────────▶ Feature Columns
                         └── numeric_column()
                         └── categorical_column()
                         └── bucketized_column()
                         └── embedding_column()
                                ↓
                     Vectorized feature representation
                                ↓
                            Model
```

## Por qué esto importa

- Los modelos no entienden texto
- Características bien representadas → modelos mejores
- Feature engineering = 50% del trabajo en ML real

## Ejemplo real completo en código

```python
import tensorflow as tf

# Definir feature columns
sqft = tf.feature_column.numeric_column("sqft")

property_type = tf.feature_column.categorical_column_with_vocabulary_list(
    "type", ["house", "apartment"]
)

property_type_one_hot = tf.feature_column.indicator_column(property_type)

# Combinar todo
feature_layer = tf.keras.layers.DenseFeatures(
    [sqft, property_type_one_hot]
)

# Modelo
model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(1)  # linear regressor
])

model.compile(optimizer="adam", loss="mse")
```
