# Cómo medir la complejidad de un modelo: L1 vs L2 Regularization

## Explicación

* La regularización es un área importante dentro del aprendizaje automático y existen varias técnicas para aplicarla.
* Su propósito principal es ayudar a que un modelo generalice mejor, es decir, que funcione bien no solo con los datos de entrenamiento sino también con datos nuevos nunca antes vistos.
* Entre las técnicas existentes, una categoría importante son las penalizaciones aplicadas al tamaño de los parámetros del modelo, conocidas como *parameter norm penalties*.

### El problema

* Un modelo demasiado complejo tiende a sobreajustarse (memoriza el entrenamiento).
* La regularización introduce una penalización adicional durante el entrenamiento para evitar que los pesos crezcan demasiado y aumenten la complejidad del modelo.

### Cómo medir la complejidad

* Una forma común de medir la complejidad del modelo es observar la magnitud del vector de pesos.
* Esa magnitud se calcula usando normas matemáticas, principalmente la norma L1 y la norma L2.

### Norma L2

* La norma L2 mide la magnitud del vector usando la distancia euclidiana.
* Matemáticamente, equivale a la raíz cuadrada de la suma de los pesos al cuadrado.
* Cuando aplicamos regularización L2, mantenemos los valores de los pesos dentro de una región con forma de círculo alrededor del origen.
* En machine learning se usa normalmente la versión al cuadrado de la norma L2 (para simplificar derivadas), lo que recibe el nombre de *weight decay*.
* Esta forma de regularización tiende a reducir los pesos sin obligarlos a llegar a cero.

### La importancia de lambda

* Lambda (λ) es un parámetro que controla cuánta importancia se le da a la simplicidad del modelo comparado con reducir el error del entrenamiento.
* Elegir el valor correcto de λ depende del conjunto de datos, por lo que debe ajustarse mediante experimentación o técnicas automáticas de búsqueda.

### Norma L1

* La norma L1 mide la magnitud como la suma del valor absoluto de los pesos.
* En este caso, la región que limita los valores permitidos tiene forma de diamante, en lugar del círculo suave de L2.
* Debido a esa forma geométrica, algunas soluciones óptimas con L1 regularización terminan con pesos exactamente en cero.

### Diferencia clave entre L1 y L2

* L1 genera modelos más *esparsos* (con muchos pesos igual a cero).
* Esto permite usar L1 como método automático de selección de características, porque los pesos en cero indican qué variables pueden eliminarse sin afectar al rendimiento.

## Resumen final en una frase

> La regularización L1 y L2 controlan la complejidad del modelo penalizando el tamaño de los pesos, donde L2 reduce los pesos suavemente sin llevarlos a cero y L1 elimina algunos completamente, funcionando además como un mecanismo de selección de características.
