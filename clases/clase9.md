---
title: No9 - PINNs (continuación)
---

# Physics-Informed Neural Networks (PINNs): restricciones y dificultades

**Fecha:** 11/05/2026

:::{iframe} https://www.youtube.com/embed/blahblahblah
:width: 100%
:::

:::{seealso}
Esta clase continúa directamente desde la [Clase 8](clase8.md), donde se introdujeron las PINNs.
:::

## Tres enfoques para incorporar ecuaciones diferenciales

Cuando queremos estimar o aproximar una ecuación diferencial con modelos de aprendizaje automático, hay tres enfoques principales:

1. **Emuladores:** se estima la ecuación diferencial con otro modelo de ML.
Por ejemplo, cuando hay ecuaciones costosas (Navier-Stokes, etc.)

2. **Restricciones suaves** (vía Lagrangiano): el modelo se entrena minimizando una función de costo empírica $\mathcal{L}_{\text{emp}}$ más un término de penalización $D[x(\theta)]$ que _incentiva_ satisfacer la ecuación diferencial.
Por ejemplo, las {term}`PINN`s:

$$
\min_{\theta,\, x} \mathcal{L}_{\text{emp}}(y, x, \theta) + \lambda \| D[x(\theta)] \|.
$$

3. **Restricciones fuertes** (se satisfacen _estrictamente_): el modelo debe satisfacer la ecuación diferencial $D[x(\theta)]$ en todo momento.
Son las **NODE** y **UDE**:

$$
\min_{\theta,\, x} \mathcal{L}_{\text{emp}}(y, x(\theta), \theta) \quad \text{s.a. } D[x(\theta)] = 0.
$$

## Dualidad Lagrangiana

Un resultado fundamental de la optimización con restricciones establece que, si relajamos la restricción fuerte $D[x(\theta)] = 0$ por una restricción aproximada $\| D[x(\theta)] \| \leq \varepsilon$, existe un $\lambda(\varepsilon)$ tal que la solución del problema con restricción suave (penalizado con $\lambda(\varepsilon)$) coincide con la solución del problema con restricción fuerte relajada:

$$
\lambda(\varepsilon) = \lambda \quad \Longleftrightarrow \quad \text{restricción suave} = \text{restricción fuerte relajada}
$$

:::{note} Hard PINN
Elegir $\lambda \to \infty$ en la formulación suave es equivalente a hacer $\varepsilon \to 0$, es decir, a forzar la restricción estrictamente.
Las Hard PINNs imponen la ecuación diferencial de forma exacta reparametrizando la solución directamente, en lugar de penalizarla.
:::

## Dificultades al estimar PINNs

En la práctica, entrenar PINNs tiene problemas importantes de optimización.
La función de pérdida tiene dos términos muy distintos: $\mathcal{L}_{\text{emp}}$ (ajuste a los datos) y $\lambda \|D[x(\theta)]\|$ (residuo de la ecuación diferencial).
Cuando $\lambda$ es muy grande, el problema se vuelve **mal condicionado**: los gradientes del término de penalización dominan, entonces el optimizador tiene problemas para elegir la dirección óptima.

```{figure} ./figures/no9_pinns_mal_condicionadas.png
:width: 75%
:align: center

Curvas de nivel de $\mathcal{L}_{\text{PINN}}$ (rosa) alrededor de la variedad donde $D[x(\cdot,\theta)]=0$ (negro). En un entorno de $D[x(\cdot,\theta)]=0$ es probable que la PINN esté mal condicionada. 
```

:::{note} Condición de descenso de gradiente
Una forma de cuantificar cuán mal condicionado está el problema de optimización es analizar el {term}`número de condición <Número de condición>` de la matriz Hessiana $H = \nabla^2 \mathcal{L}$:

$$
\kappa(H) = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}
$$

Un $\kappa(H)$ grande indica que la función de pérdida tiene direcciones con curvatura muy alta y otras con curvatura muy baja, lo que se traduce en curvas de nivel elongadas (celeste en el diagrama).
:::

### Selección de $\lambda$

En general, el vector de hiperparámetros $\lambda = (\lambda_1, \ldots, \lambda_n)$ controla cuánto pesa cada restricción.
Hay dos estrategias para elegirlos:

* **Adimensionalización:** normalizar todos los términos para que operen en la misma escala:

$$
\mathcal{L}_{\text{emp}}^k \sim \lambda_1^k \mathcal{L} \sim \lambda_2^k \mathcal{L} \sim \cdots \sim \lambda_n^k \mathcal{L}
$$

* **Elección democrática** (hiperparámetros adaptativos): igualar las normas de los gradientes de cada término, de forma que ningún $\lambda$ domine el gradiente total:

$$
\| \nabla \mathcal{L}_{\text{emp}}^k \| \sim \| \lambda_1^k \nabla \mathcal{L} \| \sim \| \lambda_2^k \nabla \mathcal{L} \| \sim \cdots
$$

## Puntos de colocación


<!-- GUIDO: ANOTÉ ESTO PERO NO ESTOY SEGURO QUE ESTÉ BIEN -->
<!-- PENSABA REVISARLO UNA VEZ QUE SE SUBAN LAS GRABACIONES -->
<!-- En una PINN hay que elegir dos tipos de puntos en el dominio temporal: -->
<!-- * **Puntos de condición inicial y de borde:** donde se impone $u(t_0) = u_0$.
* **Puntos de colocación:** donde se evalúa el residuo de la ecuación diferencial $D[x(\theta)]$. Son los puntos donde le "pedimos" al modelo que satisfaga la dinámica. -->

:::{tip} Forzar la condición inicial exactamente
Para _forzar_ exactamente la condición inicial $u_0$, se puede reparametrizar la salida de la red neuronal como:

$$
u(t) = \text{NN}(t)(t - t_0) + u_0
$$

De esta manera, en $t = t_0$ se tiene $u(t_0) = \text{NN}(t_0) \cdot 0 + u_0 = u_0$ independientemente de los parámetros de la red.
:::

## Implementación

:::{note}
Se aplica escalado a la red porque las redes con bias tienden a ajustar mejor funciones de alta frecuencia que de baja frecuencia.
El escalado busca corregir este {term}`sesgo espectral <Sesgo espectral>`.
:::
