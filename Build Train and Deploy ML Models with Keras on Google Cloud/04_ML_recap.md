# ML Recap (Resumen del flujo de Machine Learning)

- Antes de escribir una línea de código ML, hay una secuencia completa en todo proyecto:

```bash
 Definir problema → recolectar datos → preparar datos → entrenar → evaluar → desplegar → monitorear
```

- Esto puede hacerse:

- Manualmente
- O mediante una pipeline automatizada (MLOps)

## Flujo completo del ciclo de vida de ML

| Paso                    | Explicación                                 | Ejemplo                       |
| ----------------------- | ------------------------------------------- | ----------------------------- |
| 1. Data extraction  | Obtener datos desde bases, archivos, APIs   | CSV, BigQuery, imágenes       |
| 2. Data analysis    | Entender datos, detectar errores, outliers  | Exploración, estadística      |
| 3. Data preparation | Limpiar, transformar, normalizar, codificar | Min-max scaling, tokenization |
| 4. Model training   | Entrenar modelo con gradient descent    | Keras `.fit()`                |
| 5. Model evaluation | Probar rendimiento con datos no vistos      | accuracy, RMSE                |
| 6. Model validation | Validar que el modelo cumple estándar       | comparación con baseline      |
| 7. Model serving    | Hacer disponible para predicciones          | API REST, TensorFlow Serving  |
| 8. Model monitoring | Vigilar el modelo en producción             | drift, latencia, errores      |

Esto refleja el concepto de:

> ML = ciclo continuo. No es entrenar una vez y ya.

## Dos fases del ML

| Fase                       | Qué hace                               | Cuándo se usa                   |
| -------------------------- | -------------------------------------- | ------------------------------- |
| Training               | Aprende los patrones (ajusta pesos)    | Solo en desarrollo / retraining |
| Inference (Prediction) | Usa el modelo entrenado para responder | En producción                   |

Ejemplo real:

- Entrenas un modelo de traducción en Google Cloud → training
- Lo usas en tu teléfono offline → inference

## Importancia de los Features

> “Better features → faster training → more accurate predictions.”

- Una red neuronal no entiende:
  - textos
  - categorías
  - imágenes crudas (sin transformaciones)
  - audios

- Todo debe convertirse a vectores numéricos reales.

Ejemplos:

| Tipo de dato | Transformación a feature      |
| ------------ | ----------------------------- |
| Texto        | Tokenizer → Embeddings        |
| Categorías   | One-hot encoding o embeddings |
| Imágenes     | Matriz de pixels normalizada  |
| Fechas       | Extraer día, mes, día_semana  |

## Por qué las data pipelines son cruciales

- En entrenamiento esto ocurre repetidamente:

```bash
1. Abrir archivo (si no está en memoria)
2. Leer entrada
3. Usar datos para entrenamiento
```

- Si esto se hace ineficientemente → el GPU queda esperando datos, y entrenar puede ser lento.
- Por eso existe:

### `tf.data` → API para construir pipelines eficientes

- `tf.data` ayuda a:

| Acción                               | Ejemplo                       |
| ------------------------------------ | ----------------------------- |
| Leer grandes datasets                | directorios, CSV, TFRecord    |
| Transformar el dataset               | map(), batch(), shuffle()     |
| Preparar batches automáticamente     | batch size dinámico           |
| Mezclar y paralelizar para velocidad | `num_parallel_calls=AUTOTUNE` |

- Ejemplo pipeline imagen

```python
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_and_preprocess)
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
```

- Esto crea una pipeline eficiente lista para entrenamiento.

- Ejemplo pipeline texto

```python
dataset = tf.data.TextLineDataset("reviews.txt")
dataset = dataset.map(tokenize).batch(64)
```

## ¿Por qué esto importa en producción?

| Sin pipeline                | Con `tf.data`                |
| --------------------------- | ---------------------------- |
| Modelo espera datos → lento | GPU siempre ocupada → rápido |
| Difícil paralelizar         | Escala fácilmente            |
| Mucho código manual         | Código compacto y reusable   |

## Resumen

| Concepto              | Significado                                      |
| --------------------- | ------------------------------------------------ |
| Training vs inference | Aprender vs usar                                 |
| Feature engineering   | Transformar datos a números útiles               |
| Data pipeline         | Flujo automatizado de transformación             |
| `tf.data`             | Herramienta TensorFlow para pipelines eficientes |
| Batch                 | Agrupar varios ejemplos para entrenar            |
| Shuffle               | Evitar que el modelo aprenda orden incorrecto    |
