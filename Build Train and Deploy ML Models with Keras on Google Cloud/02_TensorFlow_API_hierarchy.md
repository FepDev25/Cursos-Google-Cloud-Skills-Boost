# Jerarquía de API en TensorFlow

- Pensar en TensorFlow como un edificio con varios pisos:

| Nivel      | Nombre                             | Quién usa esto                    | Dificultad     |
| ---------- | ---------------------------------- | --------------------------------- | -------------- |
| Nivel 1 | Hardware-level kernels             | Fabricantes de chips (TPUs, GPUs) | Muy difícil |
| Nivel 2 | TensorFlow C++ API (Custom ops)    | Investigadores / Muy avanzados    | Difícil    |
| Nivel 3 | Core Python API (`tf.*`)           | Programadores ML intermedios      | Medio       |
| Nivel 4 | TF Layers / Metrics / Losses       | Quien arma redes personalizadas   | Fácil       |
| Nivel 5 | Alto nivel: **Keras + Estimators** | 99% de usuarios en producción     | Súper fácil  |

## Nivel 1: Hardware Abstraction

- Código en bajo nivel que permite que TensorFlow funcione en CPUs, GPUs, TPUs.
- Tú nunca programas aquí, a menos que trabajes para NVIDIA, Google o AMD.

## Nivel 2: TensorFlow C++ API (Custom Ops)

- Si necesitas una operación matemática nueva que TensorFlow no tiene la implementas en C++.
- Si inventas un nuevo tipo de activación o convolución para investigación.
  - Flujo sería:
    - Escribes operación en C++
    - La registras en TensorFlow
    - TensorFlow genera el wrapper Python

## Nivel 3: Core TensorFlow Python API

- Aquí trabajas con:
  - tf.constant(), tf.Variable()
  - Matrices y tensores
  - tf.matmul, tf.reshape, tf.cast
- Es como usar TensorFlow solo como biblioteca matemática.

## Nivel 4: Componentes de alto nivel pero todavía flexibles

- Aquí aparecen módulos como:
  - tf.layers: Crear capas de redes neuronales
  - tf.metrics: Calcular métricas (accuracy, RMSE, etc.)
  - tf.losses: Pérdidas (cross entropy, MSE, etc.)
- Ejemplo:
  - layer = tf.keras.layers.Dense(64, activation='relu')
- Aquí ya no peleas con tensores manualmente: TensorFlow lo abstrae.

## Nivel 5: Keras / Estimator API (Más alto nivel)

- Este es el nivel recomendado en la industria. Permite:
  - Definir el modelo
  - Compilarlo
  - Entrenarlo
  - Evaluarlo
  - Guardarlo
  - Servirlo en producción
  - Con muy pocas líneas.
- Ejemplo:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
```

## TensorFlow + Cloud AI Platform (Vertex AI)

- Cloud AI Platform corta verticalmente toda la jerarquía.
- Significa que, sin importar si usas bajo nivel o alto nivel:
  - Puedes correr tu modelo en Google Cloud
  - Escalarlo a múltiples GPUs/TPUs
  - Publicarlo para inferencia como servicio

## Resumen

- La API está diseñada para que puedas elegir tu nivel de control.
  - Keras es el estándar recomendado para producción.
  - No vale la pena escribir modelos manuales usando sessions o bajo nivel.
  - Google Cloud facilita entrenamiento distribuido y deployment.
