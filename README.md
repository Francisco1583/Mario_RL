Deep Reinforcement Learning for Super Mario Bros using Double DQN

Este repositorio implementa un agente de aprendizaje por refuerzo profundo capaz de jugar de manera autónoma el nivel SuperMarioBros-1-1-v0 utilizando una arquitectura basada en Double Deep Q-Network (Double DQN). El proyecto incluye un pipeline completo de interacción con el entorno, preprocesamiento visual, almacenamiento de experiencias, entrenamiento del agente y generación de métricas de desempeño.

La solución se basa en técnicas utilizadas originalmente en Atari DRL research, adaptadas al entorno de Super Mario Bros con un conjunto reducido de acciones para simplificar el problema sin sacrificar complejidad.

Características principales

Implementación completa de Double DQN.

Red neuronal convolucional optimizada para procesar imágenes en escala de grises de 84×84.

Replay Buffer para descorrelación de experiencias.

Frame skipping, redimensionamiento y frame stacking.

Registro de métricas clave: recompensa, pérdida, duración de episodios y valor Q promedio.

Guardado y carga de checkpoints.

Scripts separados para entrenamiento e inferencia.

1. Instalación del entorno

Este proyecto depende de versiones específicas de librerías debido a compatibilidad con gym-super-mario-bros y nes-py. Antes de instalar dependencias, es obligatorio crear un entorno virtual.

Crear entorno virtual compatible
rm -rf mario-env
python3 -m venv mario-env
source mario-env/bin/activate

Actualizar herramientas base
pip install --upgrade pip setuptools wheel

Instalar dependencias numéricas
pip install numpy==1.23.5 scipy==1.12.0
pip install torch torchvision

Instalar entorno de Mario
pip install gym==0.17.2
pip install pyglet==1.5.0
pip install nes-py==8.1.1 gym-super-mario-bros==7.3.0

Otras dependencias
pip install opencv-python pillow tqdm cloudpickle future
pip install "scikit-image<0.21" imageio networkx
pip install matplotlib

2. Estructura del repositorio
.
│── agent.py          # Implementación del agente Double DQN
│── neural.py         # Red neuronal convolucional (CNN)
│── wrappers.py       # Wrappers de preprocesamiento del entorno
│── replay.py         # Replay Buffer y carga de checkpoints
│── metrics.py        # Registro y graficación de métricas
│── main.py           # Script principal de entrenamiento
│── README.md         # Documentación

3. Descripción técnica
3.1 Pipeline de preprocesamiento

El entorno se transforma mediante wrappers:

Conversión a escala de grises.

Redimensionamiento a 84×84.

Salto de frames (frame skipping).

Normalización de valores de píxeles.

Stacking de 4 frames para capturar dinámica temporal.

Estas transformaciones se implementan en wrappers.py.

3.2 Arquitectura del agente
Modelo Double DQN

El agente utiliza dos redes:

Red online: aprende continuamente y selecciona acciones.

Red target: se mantiene congelada y se sincroniza periódicamente para estabilizar el aprendizaje.

Este desacoplamiento reduce la sobreestimación de valores Q.

Red neuronal convolucional

Implementada en neural.py, con la siguiente estructura:

Conv1: 32 filtros, kernel 8×8, stride 4

Conv2: 64 filtros, kernel 4×4, stride 2

Conv3: 64 filtros, kernel 3×3, stride 1

FC: 512 unidades ReLU

Output: número de acciones discretas

Replay Buffer

Contenido en replay.py:

Almacena hasta 100,000 transiciones.

Permite muestreo aleatorio para romper correlación temporal.

Soporta warmup antes de iniciar entrenamiento.

4. Entrenamiento

El script principal es:

python main.py


Incluye:

Política epsilon-greedy con decaimiento gradual.

Warmup de 10,000 pasos antes de entrenar.

Synchronización periódica de la red target.

Guardado automático de checkpoints.

5. Inferencia y demostración

Para ejecutar un episodio usando un modelo entrenado:

python replay.py


Este script carga el checkpoint más reciente y ejecuta la política de manera greedy.

6. Métricas y visualización

El sistema registra:

Recompensa acumulada por episodio

Longitud del episodio

Pérdida de entrenamiento

Valor Q promedio

Las métricas se guardan automáticamente y se pueden graficar mediante metrics.py.

7. Resultados obtenidos

Basado en el entrenamiento realizado:

La recompensa promedio muestra una tendencia ascendente estable.

La duración de los episodios se mantiene en un rango consistente.

La pérdida se estabiliza después de un aumento inicial, como es típico en RL.

El valor Q promedio converge a un rango estable sin explosiones numéricas.

El agente logra avanzar una parte significativa del nivel y exhibe comportamientos coherentes como saltos oportunos y navegación efectiva.

8. Áreas de mejora

Entrenar durante más tiempo para permitir estrategias más avanzadas.

Explorar arquitecturas alternativas como Dueling DQN o Prioritized Replay.

Extender el espacio de acciones para controlar saltos más complejos.

Implementar un scheduler más sofisticado para epsilon decay.

Probar regularización o técnicas para mejorar estabilidad del valor Q.
