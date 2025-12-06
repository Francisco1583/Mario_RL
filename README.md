# Super Mario Bros - Agente Autónomo (Double DQN)

Este repositorio contiene la implementación de un agente autónomo diseñado para jugar el nivel **SuperMarioBros-1-1** utilizando técnicas de Aprendizaje por Refuerzo Profundo, específicamente **Double Deep Q-Network (Double DQN)**.

El proyecto fue desarrollado como parte de la materia *Analítica de datos y herramientas de inteligencia artificial II*.

## Descripción

[cite_start]El objetivo de este proyecto es entrenar un agente que pueda navegar el entorno de Super Mario Bros basándose únicamente en entradas visuales (píxeles), sin conocimiento previo de la lógica interna del juego.

### Características del Agente
* [cite_start]**Modelo:** Red Neuronal Convolucional (CNN) con arquitectura Double DQN.
* [cite_start]**Entrada:** Tensor de $84\times84$ píxeles, escala de grises, stack de 4 frames consecutivos[cite: 33, 58, 60].
* [cite_start]**Acciones:** Espacio discretizado a 2 acciones: *Avanzar derecha* y *Avanzar derecha + Salto*.
* [cite_start]**Estabilización:** Uso de Replay Buffer (100,000 transiciones) y Red Target sincronizada periódicamente.

##  Instalación y Configuración

Sigue estos pasos para preparar el ambiente de desarrollo. Es necesario crear un entorno virtual y utilizar versiones específicas de las librerías para garantizar la compatibilidad con `gym-super-mario-bros`.

### 1. Preparación del Entorno Virtual

```bash
rm -rf mario-env
python3 -m venv mario-env
source mario-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install numpy==1.23.5 scipy==1.12.0
pip install torch torchvision
pip install gym==0.17.2
pip install pyglet==1.5.0
pip install nes-py==8.1.1 gym-super-mario-bros==7.3.0
pip install opencv-python pillow tqdm cloudpickle future
pip install "scikit-image<0.21" imageio networkx
pip install matplotlib
```
# 2. Descripción técnica

## 2.1 Pipeline de preprocesamiento

El entorno se transforma mediante una serie de wrappers diseñados para preparar las observaciones antes de enviarlas a la red neuronal. Las transformaciones aplicadas son:

* Conversión a escala de grises.
* Redimensionamiento a 84×84 píxeles.
* Salto de frames (*frame skipping*).
* Normalización de los valores de píxeles.
* Stacking de 4 frames consecutivos para capturar información temporal.

Estas transformaciones se implementan en el archivo `wrappers.py`.

## 2.2 Arquitectura del agente

### Modelo Double DQN
El agente utiliza dos redes para mejorar la estabilidad del aprendizaje[cite: 188]:

* **Red online:** aprende continuamente y selecciona acciones [cite: 190-192].
* **Red target:** se mantiene congelada y se sincroniza periódicamente para estabilizar el aprendizaje [cite: 193-196].

> Este desacoplamiento reduce la sobreestimación de valores Q[cite: 197].

### Red neuronal convolucional
Implementada en `neural.py`, con la siguiente estructura[cite: 156]:

* **Conv1:** 32 filtros, kernel $8\times8$, stride 4 [cite: 157-160].
* **Conv2:** 64 filtros, kernel $4\times4$, stride 2 [cite: 165-167].
* **Conv3:** 64 filtros, kernel $3\times3$, stride 1 [cite: 170-172].
* **FC:** 512 unidades ReLU [cite: 179-180].
* **Output:** número de acciones discretas[cite: 183].

### Replay Buffer
Contenido en `replay.py`, este componente gestiona la memoria de experiencias [cite: 61-62]:

* Almacena hasta **100,000 transiciones**[cite: 226].
* Permite muestreo aleatorio para romper la correlación temporal entre frames consecutivos[cite: 65].
* Soporta **warmup** (llenado inicial) antes de iniciar el entrenamiento para garantizar diversidad de datos[cite: 67].

# 3. Entrenamiento

El script principal para entrenar al agente es: main.py


Incluye las siguientes características:

* Política **epsilon-greedy** con decaimiento gradual.
* **Warmup** de 10,000 pasos antes de iniciar el entrenamiento.
* **Sincronización periódica** de la red target para estabilizar el aprendizaje.
* **Guardado automático de checkpoints** durante el entrenamiento.

# 4. Inferencia y demostración

Para ejecutar un episodio usando un modelo entrenado:

python replay.py

Este script carga el checkpoint más reciente y ejecuta la política de manera greedy.


# 5. Métricas y visualización

El sistema registra las siguientes métricas durante el entrenamiento:

* **Recompensa acumulada por episodio**
* **Longitud del episodio**
* **Pérdida de entrenamiento**
* **Valor Q promedio**

Todas las métricas se guardan automáticamente y pueden ser graficadas utilizando el archivo: metrics.py

# 6. Resultados obtenidos

Basado en el entrenamiento realizado:

* La **recompensa promedio** muestra una tendencia ascendente estable.
* La **duración de los episodios** se mantiene en un rango consistente.
* La **pérdida** se estabiliza después de un aumento inicial, como es típico en aprendizaje por refuerzo.
* El **valor Q promedio** converge a un rango estable sin explosiones numéricas.
* El agente logra avanzar una parte significativa del nivel y exhibe **comportamientos coherentes** como saltos oportunos y navegación efectiva.



