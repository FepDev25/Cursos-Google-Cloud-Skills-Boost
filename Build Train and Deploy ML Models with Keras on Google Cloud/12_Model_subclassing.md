# Model Subclassing

## Explicación

- En Keras existen tres formas principales de construir modelos: Sequential, Functional API y Model Subclassing.
- El modelo Sequential es el más simple y directo. Está compuesto por una lista lineal de capas, pero solo permite una entrada y una salida y no soporta arquitecturas complejas.
- La Functional API ofrece una forma más flexible de construir modelos con múltiples entradas, múltiples salidas o topologías más avanzadas. Es la opción recomendada para la mayoría de los casos porque permite crear estructuras arbitrarias sin demasiada complejidad.
- El tercer método es Model Subclassing, que permite crear modelos completamente personalizados desde cero.

### Cuándo se debe usar

- Se utiliza cuando se requiere una personalización completa y las otras dos opciones no son suficientes.
- Es útil en investigaciones o casos avanzados donde se necesita un comportamiento del modelo fuera de lo común o no estándar.
- La elección entre estos tres métodos depende del nivel de personalización que se necesite.

### Cómo funciona

- Para usar model subclassing, se crea una clase que hereda de `tf.keras.Model`.
- Dentro del método `__init__` se definen las capas que el modelo usará.
- En el método `call` se define explícitamente el paso hacia adelante (forward pass), es decir, cómo fluye la información a través del modelo.

Ejemplo conceptual:

- Primero se definen las capas en el constructor.
- Luego, en `call`, se determina el orden en el que esas capas procesan los datos.

### Personalización adicional

- Se puede recibir parámetros personalizados, por ejemplo el número de clases, y usarlos para configurar capas dentro del modelo, como la cantidad de neuronas en la capa final.
- Esto da control total sobre el comportamiento del modelo.

### Entrenamiento

- Tanto Sequential como la Functional API y Model Subclassing pueden entrenarse usando `model.compile()` y `model.fit()`.
- Sin embargo, subclassing también permite escribir bucles de entrenamiento personalizados si se necesita control total.

### Control de comportamiento en entrenamiento o inferencia

- El método `call` puede incluir el argumento `training=True/False`.
- Algunas capas como Dropout o BatchNormalization tienen comportamientos distintos durante entrenamiento e inferencia, por lo que este control es necesario.

Ejemplo conceptual:

- Durante entrenamiento, Dropout elimina aleatoriamente unidades para evitar sobreajuste.
- Durante inferencia, Dropout no elimina unidades.

---

## Resumen final en una frase

> Model Subclassing es la opción más avanzada de Keras y permite construir modelos completamente personalizados definiendo manualmente el flujo de datos y las capas, ofreciendo el máximo control en comparación con Sequential y la Functional API.
