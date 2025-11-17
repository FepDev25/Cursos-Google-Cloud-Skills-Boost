# Introduction to tensorflow

- TensorFlow es una librería open-source para cálculo numérico de alto rendimiento, no solo para machine learning.
- Se puede usar para:
  - Redes neuronales
  - Operaciones matemáticas pesadas
  - Física (resolver ecuaciones diferenciales)
  - Computación en GPU/TPU

## DAG (Directed Acyclic Graph)

- TensorFlow ejecuta cálculos como un grafo
- En el grafo:
  - Nodos (circulitos): Operaciones matemáticas (sumar, multiplicar, softmax, etc.)
  - Aristas (líneas): Flujo de datos entre operaciones
  - Ej: input → matmul → add bias → activation → output

## Tensor

- Un tensor es simplemente un arreglo multidimensional.

| Nombre | Ejemplo | Dimensión | También llamado |
|--------|---------|-----------|-----------------|
| Escalar | 5 | 0D | Número |
| Vector | [2, 4, 6] | 1D | Arreglo |
| Matriz | [[1,2],[3,4]] | 2D | Tabla |
| Tensor 3D | Conjunto de imágenes, cubo de datos | 3D | tensor |
| Tensor 4D | Batch de imágenes con canales | 4D | Usado en deep learning |

## Ejemplos

- Si tienes una imagen 28×28 en blanco y negro, ¿qué tipo de tensor es?
  - Matríz: 2D tensor
Si tienes un lote de 32 imágenes de 28×28 y 1 canal:
  - Tensor shape: (32, 28, 28, 1) → 4D tensor

## Resumen

| Concepto | Explicación corta |
|--------|---------|
| TensorFlow | Librería para cálculo numérico y ML |
| Tensor | Arreglo n-dimensional que fluye en el modelo |
| DAG | Representación del cálculo como nodos (operaciones) y edges (datos) |
| Portabilidad | Entrenas en cloud → deploy en móvil, Web, IoT |
| Hardware | Se ejecuta en CPU, GPU, TPU |
| Nombre "TensorFlow" | Porque los tensores fluyen por el grafo |
