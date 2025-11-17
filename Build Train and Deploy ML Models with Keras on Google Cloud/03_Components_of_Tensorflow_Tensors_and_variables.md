# Components of Tensorflow: Tensors and variables

## Tensores

- Un tensor = arreglo n-dimensional de datos.
- Puede verse como:

| Tipo                | Ejemplo | Dimensiones (Rank) | Nombre        |
| ------------------- | ------- | ------------------ | ------------- |
| `3`                 | escalar | 0D                 | Tensor rank 0 |
| `[3,5,7]`           | vector  | 1D                 | Tensor rank 1 |
| `[[3,5,7],[4,6,8]]` | matrix  | 2D                 | Tensor rank 2 |
| `[2 matrices]`      | stack   | 3D                 | Tensor rank 3 |

- El shape del tensor te dice cuántos elementos hay en cada dimensión.
- Ejemplo:

```python
x = tf.constant([[3,5,7], [4,6,8]])
```

- Shape: `(2, 3)` → 2 filas × 3 columnas
- Rank: `2` → porque tiene dos dimensiones

- Para ver el shape:

```python
tf.shape(x)
```

### Cómo se crean tensores

```python
tf.constant(3)               # escalar
tf.constant([3,5,7])        # vector
tf.constant([[3,5,7],[4,6,8]])  # matriz
```

### Operaciones comunes con tensores

#### Stack (apilar para crear más dimensiones)

```python
x1 = tf.constant([2,3,4])       # shape: (3,)
x2 = tf.stack([x1, x1])         # shape: (2,3)
x3 = tf.stack([x2, x2, x2, x2]) # shape: (4,2,3)
```

- Vas agregando dimensiones literalmente apilando.

#### Slice (slicing: cortar)

- Ejemplo:

```python
y = x[:, 1]
```

- Significa:
  - `:` → toma todas las filas
  - `1` → toma solo la columna con índice 1

- De:

```bash
[[3,5,7],
 [4,6,8]]
```

- Slicing produce:

```bash
[5,6]
```

#### Reshape (reorganizar los datos manteniendo el total)

```python
tf.reshape(x, (3,2))
```

- Esto reorganiza:

```bash
[[3,5,7],
 [4,6,8]]
```

→ se vuelve:

```bash
[[3,5],
 [7,4],
 [6,8]]
```

- El número total de elementos debe ser el mismo.

## Variables

| TensorFlow constant | TensorFlow variable         |
| ------------------- | --------------------------- |
| No cambia su valor  | Puede cambiar               |
| Usado para datos    | Usado para pesos del modelo |
| `tf.constant()`     | `tf.Variable()`             |

Ejemplo:

```python
w = tf.Variable([1.0, 2.0, 3.0])
```

Puedes modificarla:

```python
w.assign([4.0,5.0,6.0])
w.assign_add([1,1,1])
w.assign_sub([2,2,2])
```

- Los pesos de una red neuronal son variables, porque se ajustan durante el entrenamiento.

## ¿Cómo aprende TensorFlow? → GradientTape

- Aquí empieza lo importante para ML:
- TensorFlow necesita saber cómo cambia la pérdida respecto a cada peso: eso es una derivada parcial.
- Proceso:

1. Forward pass: haces predicciones
2. Calculas la pérdida (error)
3. Backward pass: calculas gradientes
4. Actualizas pesos

- TensorFlow hace esto usando:

```python
tf.GradientTape()
```

- Ejemplo simple:

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x2 + 2*x + 1  # función

grad = tape.gradient(y, x)
print(grad)
```

Resultado esperado:

```bash
8.0
```

Porque:

derivada: `dy/dx = 2x + 2` → `2*3 + 2 = 8`

## Cómo funciona GradientTape por dentro

| Paso | Acción                                                                     |
| ---- | -------------------------------------------------------------------------- |
| 1    | TensorFlow registra todas las operaciones dentro del bloque `GradientTape` |
| 2    | Luego ejecuta la derivación usando reverse-mode autodiff               |
| 3    | Devuelve los gradientes para actualizar las variables                      |

- Esto es lo que permite el backpropagation.

## Resumen express tipo examen

| Concepto     | Explicación                                |
| ------------ | ------------------------------------------ |
| Tensor       | Arreglo multidimensional                   |
| Rank         | Número de dimensiones                      |
| Shape        | Tamaño en cada dimensión                   |
| Constant     | Tensor inmutable                           |
| Variable     | Tensor que puede cambiar (pesos)           |
| Stack        | Apilar tensores para crear mayor dimensión |
| Slice        | Extraer parte de un tensor                 |
| Reshape      | Reorganizar tensor                         |
| GradientTape | Calcula gradientes automáticamente         |

## Ejemplo real combinando todo

```python
import tensorflow as tf

# 1) crear datos como tensor
x = tf.constant([[3,5,7], [4,6,8]])

# 2) convertir a variable (simula pesos)
w = tf.Variable([1.0, 2.0, 3.0])

# 3) calcular gradiente
with tf.GradientTape() as tape:
    y = tf.reduce_sum(x * w)

grad = tape.gradient(y, w)
print("Gradient:", grad.numpy())
```
