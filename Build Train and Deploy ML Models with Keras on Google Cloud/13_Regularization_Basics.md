# Regularization Basics

## Explicación

- Cuando entrenamos un modelo, el objetivo principal es reducir el valor de la pérdida (loss).
- Sin embargo, a medida que entrenamos más tiempo, a veces el modelo comienza a funcionar bien con los datos de entrenamiento, pero empeora con los datos de prueba. Esto se observa cuando la curva de pérdida del entrenamiento baja, pero la del test sube.
- Este comportamiento indica sobreajuste u overfitting. El modelo no está aprendiendo patrones generalizables, sino memorizando los datos.

### Early stopping

- Una solución inicial es detener el entrenamiento antes de que ocurra el sobreajuste.
- Aunque sirve en algunos casos, no siempre es la mejor solución porque el problema puede no ser el tiempo de entrenamiento sino la complejidad del modelo.

### Rol de la regularización

- La regularización aparece como una forma de controlar la complejidad del modelo para reducir el sobreajuste.
- Un modelo demasiado complejo genera límites de decisión innecesariamente complicados.
- En el ejemplo de TensorFlow Playground, algunos patrones aparecen aunque no exista evidencia en los datos. Esto ocurre porque el modelo tiene demasiadas características (feature crosses) que generan decisiones arbitrarias.

### Ejemplo conceptual

- Si eliminamos características innecesarias, como cruces sintéticos, el modelo se vuelve más simple.
- A medida que reducimos la complejidad, el modelo se ajusta mejor a nuevos datos.
- Este equilibrio mejora la pérdida en el conjunto de prueba y evita comportamientos exagerados.

### Relación con la teoría

- La regularización sigue un principio conocido como la Navaja de Ockham: entre dos modelos posibles, debemos preferir el más simple siempre que explique adecuadamente los datos.
- Un modelo demasiado simple tampoco sirve. Por ejemplo, predecir siempre "5 dólares" para todas las carreras de taxi no tiene utilidad.
- El objetivo es encontrar un balance adecuado entre simplicidad y precisión.

### La importancia del parámetro lambda

- La regularización se controla mediante un coeficiente llamado lambda.
- Lambda define cuánto peso damos a la simplicidad del modelo frente a la precisión en los datos.
- Elegir el valor adecuado de lambda depende del problema y de los datos, por lo que debe ajustarse como cualquier otro hiperparámetro.

## Resumen final en una frase

> La regularización ayuda a evitar el sobreajuste penalizando modelos demasiado complejos y ajustando un equilibrio entre simplicidad y precisión mediante un coeficiente controlado llamado lambda.
