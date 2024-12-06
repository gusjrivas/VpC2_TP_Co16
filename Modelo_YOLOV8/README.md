# CEIA - Visión por computadora 2 - Co16 2024

### Integrantes del equipo:

* a1604 | Federico Arias Suárez | f_ariassuarez@hotmail.com
* a1618 | Myrna Lorena,Degano | myrna.l.degano@gmail.com
* a1620 | Gustavo Julián Rivas | gus.j.rivas@gmail.com


Fuente de datos:

https://universe.roboflow.com/myworkspace-iraqv/wildfire_full  
<br>
* <u>Tipo de problema a resolver</u>: Detección de objetos.  
<br>
* <u>Objetivo</u>: Detectar focos de incendio.  
<br>
* <u>Motivación</u>: La Amazonía se está quemando a un ritmo alarmante, afectando el clima, la biodiversidad y el calentamiento global.
Detectar incendios de manera temprana podría marcar la diferencia para evitar daños mayores.


* <u>Créditos</u>:  

 - **title**: Wildfire_Full Dataset
 - **type**: Open Source Dataset
 - **author**: MyWorkspace
 - **URL**: https://universe.roboflow.com/myworkspace-iraqv/wildfire_full
 - **published**: Roboflow Universe
 - **publisher**: Roboflow
 - **year**: 2024
 - **month**: oct

# Resumen del trabajo con modelos YOLO V8 para la detección de humo y fuego en imágenes forestales

En el siguiente link se encuentra el Colab de Google donde se aloja el trabajo realizado con el modelo:https://colab.research.google.com/drive/1D1NcvjO2rAR3-tPLlMhZvzfBZTAK5F8m?usp=sharing

# Introducción
Detectar humo y fuego en imágenes forestales de manera temprana es fundamental para prevenir incendios y minimizar su impacto. En este trabajo exploramos la posibilidad de aplicar el modelo YOLOv8 como una alternativa efectiva para este propósito. Este modelo se destaca por su capacidad de realizar detecciones en tiempo real con alta precisión, incluso en entornos complejos y dinámicos, lo cual lo hace adecuado para el monitoreo continuo de áreas forestales. A través de técnicas de augmentación avanzada y una arquitectura optimizada, YOLOv8 ofrece un rendimiento prometedor. Esperamos que la aplicación de este modelo nos permita evaluar si puede ser una alternativa factible para la detección temprana de incendios forestales.

# El modelo
YOLO Annotation Format

El formato de anotación YOLO se utiliza para entrenar modelos de detección de objetos en visión por computadora. Es sencillo y compatible con frameworks como Darknet, PyTorch, y TensorFlow.

Cada imagen tiene un archivo .txt con las coordenadas de las cajas delimitadoras de los objetos detectados, con la estructura: class_id x_center y_center width height, donde:

class_id: Índice de la clase del objeto (ej. 0: "perro").
x_center, y_center: Coordenadas normalizadas del centro.
width, height: Dimensiones de la caja delimitadora, normalizadas.
Un archivo classes.names contiene los nombres de las clases usadas.

YOLOv8 es la octava versión de la famosa arquitectura YOLO y sigue mejorando la precisión, velocidad y facilidad de implementación de las versiones anteriores. Esta versión también se centra en mantener un equilibrio entre eficiencia en tiempo de inferencia y precisión de detección.

# Transformaciones

Luego de la descarga del dataset se realizón las siguientes transformaciones base y con augmentación:


Transformaciones Base (base_transform)

$transforms.Resize((640,640)): $

Propósito: Redimensionar todas las imágenes al mismo tamaño (640x640 píxeles). Esto asegura que todas las imágenes tengan dimensiones consistentes antes de pasarlas por la red neuronal. Muchos modelos de deep learning, como YOLO, requieren que las imágenes tengan dimensiones específicas para poder procesarlas en batch.

$transforms.ToTensor(): $

Propósito: Conviertir la imagen en un tensor de PyTorch. Los modelos de deep learning requieren que los datos de entrada estén en formato de tensor. Además, transforma los valores de píxeles de 0-255 a un rango entre 0-1, lo cual es beneficioso para el entrenamiento.

$transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]): $

Propósito: Normaliza los valores de los píxeles utilizando las medias y desviaciones estándar de los canales RGB, calculadas a partir del conjunto de datos ImageNet. Esto ayuda a centrar los datos en torno a cero, acelerando la convergencia durante el entrenamiento y mejorando la estabilidad del proceso de entrenamiento.

Transformaciones con Augmentación (augmented_transform)

El objetivo principal de las augmentaciones es incrementar la diversidad del conjunto de datos para mejorar la capacidad del modelo de generalizar.

$transforms.RandomHorizontalFlip(p=0.5): $

Propósito: Realizar un volteo horizontal aleatorio de la imagen con una probabilidad del 50%. Esto permite al modelo aprender a reconocer objetos sin importar la dirección en la que estén orientados (izquierda o derecha). Esto es especialmente útil para detectar patrones como el humo o el fuego, que no deberían estar afectados por la orientación horizontal.

$transforms.RandomRotation(degrees=30): $

Propósito: Rotar aleatoriamente la imagen dentro de un rango de ±30 grados. Esto es útil para evitar que el modelo se vuelva demasiado dependiente de una orientación particular del objeto (por ejemplo, fuego en una inclinación fija). La idea e stratar de aumentar la robustez del modelo frente a variaciones angulares, asegurando que pueda detectar correctamente los objetos en cualquier orientación razonable.

$transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,hue=0.1): $

Propósito: Realizar cambios aleatorios en el brillo, contraste, saturación y tono de la imagen. Con esto tratamos de simular diferentes condiciones de iluminación. Dado que el fuego y el humo pueden aparecer en una amplia variedad de condiciones ambientales (día, noche, sombras, etc.), el ColorJitter ayuda a entrenar al modelo para que sea más robusto y pueda detectar patrones en una variedad de entornos lumínicos.

$transforms.RandomResizedCrop(size=(640,640),scale=(0.6,1.0),ratio=(0.75,1.33)): $

Propósito: Recortar aleatoriamente una porción de la imagen, que luego se redimensiona al tamaño especificado de 640x640 píxeles. Esto ayuda al modelo a no depender demasiado de la posición específica de los objetos en la imagen. Permite entrenar al modelo con distintas composiciones, haciendo que sea más adaptable a las variaciones espaciales y, por ende, mejore su capacidad de generalización.

$transforms.ToTensor()$ y  $transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]):$ 

Estas dos transformaciones son las mismas que se usan en base_transform y cumplen el mismo propósito. ToTensor() convierte la imagen a un tensor de PyTorch, y Normalize() ajusta los valores para garantizar que todas las entradas tengan el mismo rango y distribución de valores, lo cual facilita el entrenamiento estable del modelo.

### Resumen del Propósito de las Augmentaciones:

Las augmentaciones fueron incluídas para simular variaciones reales que el modelo podría encontrar en producción, como diferentes orientaciones, condiciones de iluminación y posiciones de los objetos. Estas variaciones las consideramos relevantes en la detección de fuego y humo, ya que estos fenómenos pueden aparecer en cualquier contexto ambiental. Por lo tanto, aplicar estas augmentaciones buscamos incrementar la capacidad del modelo de detectar correctamente el fuego y el humo bajo distintas condiciones, mejorando su rendimiento y robustez.

# Comparación de Resultados entre el Entrenamiento con y sin Augmentación de Datos con YOLOV8n

### Tabla de Resultados de ejecución sin data ausgmetation:

| Clase       | Imágenes | Instancias | Precisión (P) | Recall (R) | mAP@0.5 | mAP@0.5-0.95 |
|-------------|----------|------------|---------------|------------|---------|--------------|
| All         | 225      | 733        | 0.769         | 0.496      | 0.643   | 0.366        |
| Fire        | 79       | 279        | 0.789         | 0.455      | 0.632   | 0.333        |
| Smoke       | 210      | 454        | 0.748         | 0.537      | 0.654   | 0.399        |

### Tabla de Resultados de ejecución con data augmentation:

| Clase       | Imágenes | Instancias | Precisión (P) | Recall (R) | mAP@0.5 | mAP@0.5-0.95 |
|-------------|----------|------------|---------------|------------|---------|--------------|
| All         | 222      | 721        | 0.792         | 0.441      | 0.629   | 0.384        |
| Fire        | 68       | 234        | 0.786         | 0.393      | 0.601   | 0.341        |
| Smoke       | 213      | 487        | 0.799         | 0.489      | 0.656   | 0.427        |



En la comparativa de ambos modelos, se pueden observar algunas diferencias clave en el rendimiento:

Precisión (P): El modelo con augmentación avanzada mostró una ligera mejora en la precisión general, especialmente en la clase "smoke", que pasó de 0.748 a 0.799. Esto sugiere que la augmentación ayudó al modelo a aprender más características específicas de los objetos.

Recall (R): En general, el modelo sin augmentación tuvo un mejor recall (0.496 frente a 0.441). Sin embargo, en el caso específico de la clase "smoke", el recall del modelo con augmentación fue ligeramente más bajo, indicando que, aunque el modelo es más preciso, puede estar perdiendo algunos ejemplos positivos.

mAP: En cuanto a mAP@0.5-0.95, el modelo con augmentación avanzó de 0.366 a 0.384, y en particular, la clase "smoke" mejoró de 0.399 a 0.427. Esto sugiere una mejora en la capacidad del modelo para detectar objetos más difíciles.

En resumen, el uso de data augmentation parece mejorar la precisión del modelo, especialmente para la clase "smoke", y optimiza el rendimiento en la métrica mAP@0.5-0.95, lo cual indica una mejora en la capacidad del modelo para generalizar. No obstante, se observa una reducción en el recall, lo cual significa que el modelo es menos exhaustivo en la detección de todos los objetos presentes. Esto puede ser aceptable si se busca un equilibrio entre reducir falsos positivos y lograr una detección más precisa.

De todas maneras en algún caso se detectó nubes como humo.

# Tunning de data augmentation y prueba con YOLO V8m

El cambio del modelo YOLOv8n al modelo YOLOv8m se realizó para aprovechar la mayor capacidad del modelo YOLOv8m en comparación con YOLOv8n.

YOLOv8m tiene una arquitectura más compleja, con más parámetros y capas, lo cual le permite aprender características más ricas y complejas de los datos. Esto debería resultar en una mejor capacidad de detección, especialmente para objetos difíciles de identificar o cuando hay muchas variaciones en las imágenes.

Las principales razones para el cambio incluyen:
- **Mejora en la Precisión**: YOLOv8m tiene más capacidad para capturar patrones complejos gracias a su mayor número de parámetros, lo cual mejora la precisión general del modelo.
- **Mejor Rendimiento en Escenarios Complejos**: En situaciones donde hay alta variabilidad en las imágenes (por ejemplo, diferentes condiciones de iluminación, orientaciones y tamaños de objetos), YOLOv8m puede aprender mejor debido a su arquitectura más robusta.
- **Capacidad de Generalización**: Dado que se utilizó una augmentación avanzada y se incrementó la diversidad del conjunto de datos, un modelo más complejo como YOLOv8m tiene más capacidad de aprovechar estas augmentaciones para generalizar mejor sobre datos no vistos.

### A partir de la ejecución del modelo entrenado con el YOLOv8m y augmentación avanzada, podemos observar lo siguiente.

###  Tabla de Resultados de ejecución con data augmentation yolov8m:

| Clase       | Imágenes | Instancias | Precisión (P) | Recall (R) | mAP@0.5 | mAP@0.5-0.95 |
|-------------|----------|------------|---------------|------------|---------|--------------|
| All         | 222      | 721        | 0.799         | 0.737      | 0.821   | 0.598        |
| Fire        | 68       | 234        | 0.782         | 0.684      | 0.780   | 0.489        |
| Smoke       | 213      | 487        | 0.815         | 0.791      | 0.861   | 0.707        |


#### En conclusión, a partir de los resultados obtenidos tras entrenar el modelo YOLOv8m con data augmentation:

Mejor Precisión y Recall: Con el uso del modelo YOLOv8m y técnicas avanzadas de augmentación, se lograron mejoras notables en las métricas clave como la precisión (P) y el recall (R). La clase "smoke" mostró resultados especialmente sólidos, con un recall del 79.1%, lo cual sugiere que el modelo tiene buena capacidad para detectar humo en diferentes condiciones.

Mayor mAP@0.5 y mAP@0.5-0.95: En comparación con los modelos anteriores (YOLOv8 sin augmentación y YOLOv8 con augmentación), el uso de YOLOv8m resultó en un aumento considerable de mAP@0.5 (82.1%) y mAP@0.5-0.95 (59.8%). Esto indica que el modelo mejora su capacidad general de detección y precisión en la localización de los objetos.

Conclusión General: El modelo YOLOv8m, junto con una adecuada augmentación y un ajuste de hiperparámetros, resulta una alternativa prometedora para la detección de humo y fuego en imágenes forestales, mostrando mejoras significativas tanto en la capacidad de detección como en la generalización en condiciones variadas y complejas.

# Nuevo tunning aplicado a YOLOv8m

 A partir de la ejecución del modelo entrenado con el YOLOv8m con data augmentation avanzada y se realizó un nuevo tunning para mejorar la precisión y recall.

###  Tabla de Resultados de ejecución con data augmentation yolov8m:

| Clase       | Imágenes | Instancias | Precisión (P) | Recall (R) | mAP@0.5 | mAP@0.5-0.95 |
|-------------|----------|------------|---------------|------------|---------|--------------|
| All         | 222      | 721        | 0.838         | 0.712      | 0.813   | 0.594        |
| Fire        | 68       | 234        | 0.829         | 0.644      | 0.767   | 0.488        |
| Smoke       | 213      | 487        | 0.846         | 0.781      | 0.860   | 0.699        |


Este resumen los resultados del último ajuste del modelo YOLOv8m con augmentación avanzada y ajuste de hiperparámetros. En general, las métricas indican una mejora significativa en comparación con ejecuciones anteriores, especialmente en la precisión (P) y el mAP para ambas clases, "fire" y "smoke".

El modelo alcanzó una precisión global de 83.8% y un recall de 71.2%, mientras que el mAP@0.5-0.95 fue de 59.4%. La clase "smoke" mostró un rendimiento destacado con una precisión de 84.6% y un recall de 78.1%, lo cual refleja que el modelo tiene una mejor capacidad de detección del humo en comparación con el fuego.

La aplicación de la augmentación y el ajuste de hiperparámetros permitieron mejorar la capacidad de generalización del modelo, mostrando resultados más robustos y consistentes. Esto sugiere que YOLOv8m con un entrenamiento optimizado es una alternativa prometedora para la detección de humo y fuego en imágenes forestales, contribuyendo potencialmente a la prevención de incendios.

# Comparativa con nuevo ajuste de hiperparámetros

Se realizaron nuevos ajuste de hiperparametros para ver si se lograba obtener un mejor rendimiento.

### tabla de Resultados de ejecución con data augmentation yolov8m:

| Clase       | Imágenes | Instancias | Precisión (P) | Recall (R) | mAP@0.5 | mAP@0.5-0.95 |
|-------------|----------|------------|---------------|------------|---------|--------------|
| All         | 222      | 721        | 0.588         | 0.579      | 0.589   | 0.336        |
| Fire        | 68       | 234        | 0.560         | 0.573      | 0.571   | 0.304        |
| Smoke       | 213      | 487        | 0.617         | 0.585      | 0.607   | 0.367        |

Al comparar los resultados del ajuste reciente con los del ajuste anterior, se observa una caída significativa en las métricas clave, como la precisión (P), el recall (R), y el mAP@0.5-0.95. La precisión promedio pasó de 79.9% a 58.8%, mientras que el recall disminuyó de 73.7% a 57.9%. El mAP@0.5 también mostró una reducción de 82.1% a 58.9%, lo cual indica que el rendimiento global del modelo se vio afectado.

Para ambas clases ("fire" y "smoke"), se observó una disminución en precisión y recall. En el ajuste anterior, la clase "smoke" presentaba una buena combinación de precisión y recall, con un mAP@0.5-0.95 de 70.7%, mientras que en el ajuste más reciente, el mAP@0.5-0.95 cayó a 36.7%. Estos resultados sugieren que la configuración actual de hiperparámetros y el enfoque de augmentación no lograron el mismo nivel de efectividad que el ajuste previo.


# Conclusión General

E mejor modelo logrado en el trasncurso de este trabajo con modelos yolo v8 es el modelo que alcanzó una precisión global de 83.8% y un recall de 71.2%, mostrando mejoras significativas tanto en la capacidad de detección como en la generalización en condiciones variadas y complejas.








