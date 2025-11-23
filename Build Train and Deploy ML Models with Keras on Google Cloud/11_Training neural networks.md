# Training neural networks with TensorFlow 2 and the Keras Functional API

## Explicación

- Cuando entrenamos redes neuronales, queremos que el modelo no solo aprenda datos exactos del entrenamiento, sino que también pueda reconocer patrones nuevos que nunca antes vio.
- Esto se parece mucho a cómo aprendemos los humanos.
- Por ejemplo, si vemos varias aves como gaviotas o palomas que pueden volar, nuestro cerebro crea una regla general: “las aves con alas vuelan”.
- Sin embargo, con el tiempo también aprendemos excepciones como pingüinos o avestruces, y estas excepciones quedan guardadas como memoria específica.
- En Machine Learning pasa algo parecido. Existen modelos que son muy buenos para memorizar información específica (como reglas exactas), y otros que son mejores para generalizar, es decir, encontrar patrones y aplicarlos a datos nuevos.
- Una técnica moderna combina ambas capacidades en un solo modelo, lo cual es muy útil cuando trabajamos con datos grandes y variados, como sistemas de recomendación, clasificación o búsqueda.

### ¿Por qué necesitamos esto?

Muchos tipos de datos, especialmente los categóricos (por ejemplo: país, usuario, categoría de producto, palabras, etc.), pueden tener miles de valores distintos, pero en un registro solo se usa uno. Esto hace que la representación de los datos sea muy amplia, con muchas columnas que están vacías para la mayoría de filas.

Este tipo de estructura se conoce como una matriz dispersa (sparse matrix). Para trabajar con algo así, necesitamos modelos capaces de:

- simplificar y reducir esa información
- aprender qué combinaciones son importantes
- tomar decisiones aunque falten datos o aparezcan nuevos valores

### ¿Cómo se hace con TensorFlow y Keras?

Para crear modelos complejos en los que hay varias entradas, varias salidas o diferentes rutas internas del modelo, se utiliza la API Funcional de Keras.

Esta API permite construir modelos conectando capas como si fueran bloques o nodos de un grafo. Esto ayuda a diseñar estructuras flexibles como:

- modelos con varias entradas (texto + imagen + números)
- modelos con varias salidas (por ejemplo: clasificar y también resumir texto)
- modelos que mezclan partes anchas (memorizar) con partes profundas (generalizar)
- modelos donde se reutilizan las mismas capas para diferentes entradas (compartición de pesos)

Esto no sería posible con el modelo secuencial tradicional porque ese solo permite una arquitectura lineal (capa → capa → capa).

### Entrenar y usar el modelo

Algo importante es que, una vez construido el modelo con la API Funcional, entrenarlo, evaluarlo e inferir predicciones se hace igual que con un modelo simple. Así, la dificultad está en diseñarlo, no en entrenarlo.

### Ventajas

- Permite detectar errores antes de entrenar.
- Facilita visualizar la arquitectura del modelo.
- Se puede guardar o reutilizar el modelo sin necesidad del código original.

### Limitaciones

- No todas las arquitecturas pueden construirse con esta API.
- Si el modelo necesita estructuras dinámicas o comportamientos personalizados, es necesario usar clases personalizadas mediante subclassing.

## Resumen final en una frase

> La API Funcional de Keras permite construir modelos complejos y flexibles que combinan memorizar y generalizar, admiten múltiples entradas y salidas, y son ideales para tareas modernas como recomendadores, clasificación avanzada o procesamiento de datos diversos.
