# Activación en Redes Neuronales: ¿Por qué son necesarias?

- Las funciones de activación son esenciales en las redes neuronales profundas porque permiten que el modelo aprenda relaciones no lineales. Sin ellas, no importa cuántas capas agregues: la red se comportará como un modelo lineal.

## 1. Recordando el modelo lineal

- Un modelo lineal típico luce así:

```bash
Y = W1*X1 + W2*X2 + W3*X3 + (bias)
```

- Si agregamos una capa oculta sin activación, matemáticamente sigue siendo equivalente a una sola transformación lineal.
- Incluso si añades más capas sin funciones de activación, estas se pueden colapsar en una sola matriz multiplicada por la entrada original.
- Conclusión: Una red profunda sin funciones de activación no aprende nada mejor que una regresión lineal.

## 2. ¿Cómo rompemos esa linealidad?

- Introduciendo no linealidad, usando funciones como:
  - Sigmoid
  - tanh
  - ReLU
- Después del cálculo lineal ( W·X + b ), pasamos ese resultado por una función de activación:

```bash
Output = Activación(W·X + b)
```

- Esto transforma cada neurona en un pequeño bloque capaz de aprender patrones complejos, no solo líneas rectas.

## 3. Cómo se usa en una red

- La mayoría de redes neuronales siguen esta regla:

| Capa                                    | Activación                                                                                            |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Capas ocultas (todas excepto la última) | NO lineales (ReLU, tanh, etc.)                                                                        |
| Capa final depende del problema         | - Lineal → regresión br - Sigmoid → clasificación binaria br - Softmax → clasificación multiclase |

Esto evita que la red vuelva a colapsar en un modelo simple.

## Problemas en activaciones antiguas: *Vanishing Gradient*

- Funciones como sigmoid y tanh pueden “saturarse” cuando el valor es muy grande o muy pequeño.
- Esto provoca:
  - gradiente ≈ 0
  - pesos no se actualizan
  - el entrenamiento se detiene
- A este problema se le llama gradiente desaparecido (vanishing gradient).

## ReLU: La activación moderna por defecto

- La función ReLU (Rectified Linear Unit) se define así:

```bash
ReLU(x) = 0 si x < 0
          x si x ≥ 0
```

- Ventajas:
  - Simple
  - No se satura en valores grandes
  - Entrena hasta 10 veces más rápido que sigmoid en redes profundas
- Sin embargo, tiene una desventaja importante.

## Problema: “Dying ReLU”

- Si muchos valores entran en la zona negativa, la salida es cero, el gradiente es cero, y la neurona deja de aprender: Se vuelve inútil.

## Variantes modernas de ReLU

- Para solucionar el problema, existen versiones mejoradas:

| Activación                  | Idea principal                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------ |
| Softplus                | Suaviza el borde de ReLU (continuo y derivable)                                                        |
| Leaky ReLU              | Permite pequeños valores negativos, evita “neurona muerta”                                             |
| PReLU (Parametric ReLU) | La red aprende cuánto "fuga" debe haber                                                                |
| ELU                     | Usa valores exponenciales para mejorar estabilidad y acercar la media a cero                           |
| GELU                    | Función moderna usada en modelos grandes (ej: Transformers). Suaviza la activación usando probabilidad |

- Estas versiones mantienen las ventajas de ReLU con mejoras en estabilidad y velocidad de aprendizaje.

## Resumen visual mental

```bash
Entrada → Operación lineal (W·X + b) → Función de activación → Siguiente capa
```

- Sin activación → modelo lineal
- Con activación → modelo capaz de aprender patrones complejos

## Resumen final

| Concepto clave             | Explicación                                                           |
| -------------------------- | --------------------------------------------------------------------- |
| ¿Por qué activar?          | Para agregar no linealidad y permitir aprender patrones complejos     |
| ¿Qué pasa sin activación?  | La red, aunque sea profunda, se vuelve equivalente a un modelo lineal |
| Problemas con sigmoid/tanh | Gradientes desaparecen → aprendizaje lento o detenido                 |
| Por qué se usa ReLU        | Simple, rápida y eficiente                                            |
| Problema de ReLU           | Puede dejar neuronas inactivas para siempre                           |
| Alternativas               | Leaky ReLU, PReLU, ELU, GELU, Softplus                                |
