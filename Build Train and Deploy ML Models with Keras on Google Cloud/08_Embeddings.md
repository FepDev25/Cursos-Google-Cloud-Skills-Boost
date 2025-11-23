# Embeddings

## ¿Por qué necesitamos embeddings?

Recordemos el problema:

- Una feature categórica con muchos valores únicos
- Si usamos one-hot encoding, el vector es enorme, con casi todos ceros y un solo 1.

- Ejemplo extremo:

```bash
500,000 películas → one-hot = vector de 500,000 dimensiones
```

- Esto es:
  - Ineficiente en memoria
  - Computacionalmente caro
  - Imposible de aprender patrones semánticos (solo dice identidad)

- Embeddings resuelven eso.

## ¿Qué es un embedding?

Un embedding convierte una categoría en un vector denso de baja dimensión.

Ejemplo:

| Movie ID    | One-hot (500k dims) | Embedding (5 dims)                 |
| ----------- | ------------------- | ---------------------------------- |
| "Shrek"     | `[0,0,0,…,1]`       | `[0.81, -0.32, 1.44, 0.07, -0.55]` |
| "Star Wars" | `[0,0,0,…,1]`       | `[1.02, -0.11, 1.61, 0.20, -0.61]` |

- Ahora, dos películas cercanas en significado tendrán embeddings cercanos en el espacio vectorial.

## ¿Para qué sirven los embeddings?

- 3 grandes usos:

| Propósito                                   | Ejemplo                                    |
| ------------------------------------------- | ------------------------------------------ |
| ① Encontrar elementos similares             | recomendadores (Netflix, Spotify, tiendas) |
| ② Usarse como entrada en modelos supervised | clasificación, regresión, NLP              |
| ③ Visualización                             | TensorBoard → clusters (como MNIST)        |

## Embeddings como representación de significado

El ejemplo de películas lo explica con intuición:

- Si alguien ve Star Wars, probablemente le guste Batman
- Si alguien ve Bleu, probablemente le guste Memento

- Agrupar películas en un espacio de 2D o 3D permite que:
  - Distancias = similitud semántica
  - Cercanía = mejor recomendación

- Esto NO estaba explícito en los datos; el modelo lo aprende solo. Eso es lo mágico de embeddings.

## Dimensionalidad de embeddings

Tenemos dos números clave:

| Símbolo                            | Significa             |
| ---------------------------------- | --------------------- |
| N = número total de categorías | Ej: 500,000 películas |
| D = tamaño del embedding       | Hiperparámetro        |

Regla empírica del curso:

> D ≈ cuarta raíz de N

- Ejemplo:

```bash
N = 500,000 → sqrt(500,000) ≈ 700 → sqrt(700) ≈ 26
```

Entonces un embedding inicial razonable:

```bash
D ≈ 25
```

- Luego este parámetro puede ajustarse usando hyperparameter tuning.

## Embeddings vs One-Hot

| Propiedad          | One-hot                           | Embedding         |
| ------------------ | --------------------------------- | ----------------- |
| Tamaño             | Igual a ##categorías               | Mucho más pequeño |
| Valores            | 0 o 1                             | Números reales    |
| Aprende relaciones | No                              | Sí              |
| Eficiencia         | Muy mala si hay muchas categorías | Excelente         |

## ¿Cómo se implementan embeddings en TensorFlow?

- Si tienes una columna categórica:

```python
movie_id = tf.feature_column.categorical_column_with_identity(
    key="movie_id", num_buckets=500000
)
```

- La convertimos a embedding:

```python
movie_embedding = tf.feature_column.embedding_column(
    movie_id, dimension=25
)
```

- Luego se usa en un modelo:

```python
feature_layer = tf.keras.layers.DenseFeatures([movie_embedding])

model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1)
])
```

## Feature Crosses (bonus)

- Además de embeddings, existe otra técnica poderosa:
  - feature crosses = combinar features para crear relaciones no lineales

- Ejemplo real estate:

```bash
(city × property_type)
```

- "New York + Apartment" se trata como una categoría distinta de "New York + House".
- Esto NO genera todas las combinaciones manualmente (sería gigante).
- Usa hashing para mantenerlo controlado.

## Entrenamiento con embeddings

Después de definir feature columns:

1. Construyes un `tf.data` dataset
2. Usas un `DenseFeatures` layer
3. Entrenas con `model.fit()`

- Si los datos son pequeños → usa NumPy
- Si son grandes → usa `tf.data.Dataset`

## Resumen

| Pregunta                                        | Respuesta                                          |
| ----------------------------------------------- | -------------------------------------------------- |
| ¿Por qué no usar one-hot en categorías enormes? | Ineficiente y no captura similitud                 |
| ¿Qué es un embedding?                           | Vector numérico denso que representa una categoría |
| ¿Qué determina la dimensión del embedding?      | Hiperparámetro (cuarta raíz recomendada)           |
| ¿Cuál es un uso clave?                          | Recomendación, NLP, clustering                     |
| ¿Cómo se entrena el embedding?                  | Se aprende con gradiente durante entrenamiento     |
