# KNN interactivo en 2D y 3D — código comentado y explicación de cada bloque

## Prompts originales utilizados para crear y mejorar este proyecto

A continuación se recogen **literalmente, en orden cronológico**, los mensajes que me has dado durante esta conversación para crear y modificar el proyecto. Se conserva la redacción original (incluidas las faltas de ortografía).

### Prompt 1 — Crear una explicación visual de KNN

> quiero que me hagas un html de una explicacion interactiva de knn para explicarlo en clase de manera visual

### Prompt 2 — Clases, frontera de decisión y código comentado

> quiero que me dejes seleccionar el numero de clases, me marque la frontera de decisión, y no quiero que me des solo el html quiero que me des el codigo y lo que equivale tambien, y comentado con que haces en cada parte

### Prompt 3 — 2D, 3D y ejemplo práctico

> varios mejoras:
>
> 1. Quiero que puedas ponerlo tanto en formato 2D como 3D
> 2. Quiero que me añadas un ejemplo practico

### Prompt 4 — Entregar todo en un ZIP

> damelo todo comprimido en un zip

### Prompt 5 — Incorporar los prompts al README

> quiero que en el readme me pongas los prompts que te he dado

### Prompt 6 — Explicación en un clic y ajuste del diseño

> quiero que lo del apartado de los pasos me lo hagas todo en un paso dandole a un click ycambia el formato para que no aparezca este hueco en blanco:

*El mensaje iba acompañado de una captura de pantalla del espacio en blanco.*

### Prompt 7 — Círculo de vecinos considerados

> ahora quiero que le añadas un circulo con los que tiene en cuenta

---

## Qué se entrega

- `index.html`: estructura de la página, botones, selectores, leyenda, gráfico y panel de explicación.
- `styles.css`: aspecto visual, colores, espaciados, vista adaptable y estados de los botones.
- `app.js`: matemáticas KNN, conjunto Iris, generación de muestras, dibujos 2D/3D y eventos.
- `README.md` / `LEEME_codigo.md`: esta guía con los **prompts originales** y las **equivalencias entre código y matemáticas**.
- `GUION_CLASE.md`: un guion para enseñar el concepto y el ejemplo.

Abre `index.html` desde esta carpeta en Chrome, Edge o Firefox. Los tres archivos deben estar **juntos**. Alternativamente, abre `DEMO_ABRIR_EN_NAVEGADOR.html` en la carpeta principal del ZIP: incluye todo embebido y funciona sin Internet.

## 1. Interfaz HTML: ¿qué significa cada parte?

| ID del elemento | Qué hace | Código JavaScript que lo utiliza |
| --- | --- | --- |
| `freeBtn` y `irisBtn` | Cambiar entre ejemplo didáctico y flores Iris | `scenario()` |
| `view2DBtn` y `view3DBtn` | Escoger dos o tres variables | `dims()` y `distance()` |
| `classSelect` | Elegir entre 2 y 6 clases (solo en el modo libre) | `makeFree()` |
| `kSlider` | Elegir K entre 1 y 15 | `changeK()` y `predict()` |
| `areasToggle` | Pintar el área clasificada | `build2DMap()`, `draw3D()` |
| `boundaryToggle` | Dibujar la frontera entre clases | `build2DMap()`, `buildSlice()`, `buildVolume()` |
| `neighborhoodToggle` | Mostrar el círculo 2D / esfera 3D que contiene los K vecinos | `getNeighborhood()`, `drawNeighborhood2D()`, `drawNeighborhood3D()` |
| `boundary3DMode` | Elegir corte horizontal Z o superficie aproximada | `draw3D()` |
| `sliceZ` | Elegir a qué altura del eje Z ver la frontera | `buildSlice()` |
| `queryX`, `queryY`, `queryZ` | Medir/desplazar el nuevo ejemplo | `state.query` y `renderCoords()` |
| `nextStep` | Mostrar simultáneamente los tres pasos en un solo clic | `renderSteps()` |
| `knnCanvas` | Dibujar toda la visualización | `draw2D()` o `draw3D()` |

El HTML **no ejecuta KNN**: solamente define dónde aparecen los controles y qué contenidos se muestran. El CSS solamente define el diseño. Toda la lógica de clasificación está en JavaScript.

## 2. KNN: equivalencia matemática y código

### 2.1 Calcular la distancia

Dos variables, 2D:

**d(p, q) = √[(px − qx)² + (py − qy)²]**

Tres variables, 3D:

**d(p, q) = √[(px − qx)² + (py − qy)² + (pz − qz)²]**

La función de JavaScript que implementa exactamente esas dos fórmulas es:

```js
function distance(a,b) {
  const dx=a.x-b.x, dy=a.y-b.y;
  return Math.sqrt(dx*dx + dy*dy +
      (state.dims===3 ? (a.z-b.z)**2 : 0));
}
```

**Equivalencia:** `dx*dx` representa `(px−qx)²`; la parte `state.dims===3` añade `(pz−qz)²` **únicamente en 3D**. Esto demuestra que la predicción cambia realmente al usar tres características.

### 2.2 Elegir K vecinos

En matemáticas, primero calculamos `d(q, pi)` para cada ejemplo `pi`, ordenamos de menor a mayor y tomamos `K`:

```js
const nearest = sortedNeighbors(query)
  .slice(0, Math.min(state.k, state.points.length));
```

`sortedNeighbors` genera `{point, index, d}` para cada ejemplo y ordena por `d`. El `index` soluciona empates de distancia de forma reproducible.

### 2.3 Clasificación (votación mayoritaria)

**Clase(q) = argmax(c) Σ[I(clase(vecino_i) = c)]**, para `i = 1...K`.

En código: `votes[n.point.c]++` cuenta un voto por cada vecino elegido, y el bucle selecciona la clase con más votos. Cuando dos clases empatan, elige la que tenga una suma de distancias menor; si persiste el empate, queda la de menor índice. Es una regla explícita de desempate de esta demostración, no una propiedad obligatoria de todo KNN.

### 2.4 Regresión (media)

**ŷ(q) = (y₁ + y₂ + ... + yK) / K**.

```js
const value = nearest.reduce((s,n)=>s+n.point.v, 0) / nearest.length;
```

Esta opción se ofrece en el laboratorio libre. **El caso Iris siempre es clasificación**, porque queremos predecir una especie, no un número.

## 2.5 Círculo / esfera de los vecinos que tiene en cuenta KNN

El algoritmo calcula la distancia desde el dato nuevo **q** a todos los ejemplos conocidos, los ordena y elige los **K primeros**. El radio del entorno dibujado es la distancia del **último vecino elegido**:

**r_K = d(q, p_(K))**.

En **2D**, se representa la región **(x − qx)² + (y − qy)² ≤ r_K²** como un círculo violeta semitransparente. El gráfico 2D mantiene la misma escala para los dos ejes; por eso el radio no se deforma visualmente.

En **3D**, se representa **(x − qx)² + (y − qy)² + (z − qz)² ≤ r_K²** como una esfera translúcida con tres aros. Como el gráfico es una proyección a una pantalla 2D, un punto puede *parecer* estar dentro de la esfera por perspectiva y aun así encontrarse fuera en el espacio tridimensional.

```js
function getNeighborhood(query=state.query) {
  const nearest = predict(query).nearest; // K muestras exactas, ya ordenadas
  return {
    radius: nearest.length ? nearest[nearest.length-1].d : 0,
    indices: nearest.map(n => n.index),
    count: nearest.length
  };
}
```

El contorno **no es la frontera de decisión**: el contorno delimita los vecinos empleados *para un único punto de consulta*; la frontera separa regiones predichas para todos los posibles puntos. La circunferencia toca al K-ésimo vecino y podría incluir otros puntos empatados exactamente en distancia; aun así, el algoritmo cuenta **exactamente K** vecinos gracias al orden estable por índice.

En los controles puedes activar/desactivar «Mostrar círculo de los K vecinos» y el radio cambia automáticamente con **K**, al mover el dato nuevo o al cambiar entre **2D/3D**.

## 3. Cómo obtenemos la frontera de decisión

### 3.1 En 2D

`build2DMap()` divide el plano en pequeñas celdas. En el centro de cada celda aplica `predict(q)` para calcular su clase:

- **Región coloreada:** conjunto de celdas que KNN asigna a la misma clase.
- **Frontera de decisión:** un segmento entre dos celdas vecinas si sus clases predichas son distintas.

Es una **aproximación discreta** a la frontera matemática real, determinada por la resolución de la cuadrícula.

### 3.2 En 3D, corte Z

Una frontera en 3D no es una línea, sino una **superficie que separa volúmenes**. La función `buildSlice()` aplica KNN a una cuadrícula de puntos **(x, y, z fijo)**; los cambios entre clases se dibujan como líneas sobre ese plano. Mueve el control `Plano Z` y observarás cómo cambia la sección de la frontera.

Es el método más fácil de explicar en clase porque permite comparar la imagen con la frontera de 2D.

### 3.3 En 3D, superficie aproximada

`buildVolume()` discretiza el espacio en una malla de `14 × 14 × 14` voxeles. Aplica KNN en el centro de cada voxel y añade una cara donde hay una clase distinta en dos voxeles vecinos. Esto crea una **aproximación volumétrica** de las superficies de decisión (no una superficie continua exacta). El código de `project()` convierte posiciones 3D en píxeles 2D usando ángulos de cámara; la cámara puede girarse con el ratón.

## 4. Ejemplo práctico: identificación de flores Iris

El dataset clásico **Iris** contiene medidas de flores de tres especies. Para mantener la visualización legible hemos incluido **45 flores reales**: 15 `setosa`, 15 `versicolor`, 15 `virginica`, extraídas del conjunto original de 150.

**Las variables de entrada son:**

- **X:** longitud del pétalo (centímetros).
- **Y:** anchura del pétalo (centímetros).
- **Z:** longitud del sépalo (centímetros), utilizada al activar 3D.

**Salida del algoritmo:** una de las tres especies.

El **nuevo ejemplo es hipotético**, con tres medidas elegidas para la demostración: pétalo `4,7 cm` de largo, `1,4 cm` de ancho y sépalo `6,1 cm` de largo. No debe confundirse con una observación reservada como prueba. No se está midiendo la exactitud estadística del modelo.

### ¿Por qué normalizar?

Si dos características tienen unidades o escalas distintas, la mayor escala puede dominar la distancia euclídea. Aquí aplicamos transformación min-max usando los rangos de Iris:

**x_normalizado = 100 × (x − mínimo) / (máximo − mínimo)**.

Por ejemplo, para una longitud de pétalo de 4,7 cm, rango 1,0 a 6,9 cm:

**X = 100 × (4,7 − 1,0) / (6,9 − 1,0) ≈ 62,7**.

Se realiza con `normalize(value,i)`, y `denormalize(value,i)` permite mostrar las medidas originales en los deslizadores. Para una aplicación real, los mínimos y máximos deben estimarse **solo en el conjunto de entrenamiento**, evitando fugas de información.

Fuente del dataset clásico: [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) y `scikit-learn`, `load_iris()`.

## 5. Índice de funciones (app.js)

| Bloque | Función | Explicación |
| --- | --- | --- |
| A | `normalize`, `denormalize` | Convertir medidas reales a escala 0–100 y viceversa |
| B | `distance` | Distancia euclídea 2D/3D |
| B | `sortedNeighbors` | Obtener la lista ordenada de vecinos |
| B | `predict` | Clasificación por votos o regresión por media |
| B | `getNeighborhood` | Distancia hasta el K-ésimo vecino e índices exactos de vecinos |
| D | `drawNeighborhood2D` | Pintar el círculo de vecinos usando la escala matemática real |
| E | `drawNeighborhood3D` | Pintar la esfera de vecinos y los aros de referencia |
| C | `makeFree` | Generar ejemplos didácticos tridimensionales |
| C | `makeIris` | Convertir 45 flores a puntos con su clase |
| D | `to2D`, `from2D` | Convertir unidades del gráfico y píxeles |
| D | `build2DMap` | Calcular regiones y fronteras 2D |
| E | `project` | Girar y proyectar el espacio 3D a la pantalla |
| E | `buildSlice` | Dibujar regiones y fronteras en un plano Z |
| E | `buildVolume` | Aproximar superficies 3D entre voxeles de clases distintas |
| E | `draw3D` | Mostrar puntos, ejes y superficies con profundidad |
| F | `renderSteps` | Mostrar cálculo, vecinos y votos |
| G | `scenario`, `dims`, `changeK` | Procesar cambios de controles |

## 6. Comprobación desde la consola

En Chrome pulsa **F12 → Consola** y escribe:

```js
KNN_DEMO.state.dims                 // 2 o 3
KNN_DEMO.state.classCount           // 2 a 6 en libre; 3 en Iris
KNN_DEMO.state.points.length        // 21 puntos iniciales; 45 en Iris
KNN_DEMO.predict(KNN_DEMO.state.query)  // Resultado + K vecinos + votos
KNN_DEMO.dims(3)                    // Activar el cálculo 3D
```

**Limitación técnica intencionada:** las fronteras se estiman en una malla finita para que la animación funcione sin instalar librerías ni conectarse a un servidor. Cambiar la malla o su densidad cambia la precisión visual, pero el KNN de cada punto sigue utilizando su distancia real dentro del sistema de coordenadas normalizado.
