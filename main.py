import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import datetime
import random
from pathlib import Path

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from agent import Mario
from metrics import MetricLogger
from wrappers import ResizeObservation, SkipFrame



# Inicializar entorno de Super Mario Bros

def build_environment():
    # Crear entorno base
    environment = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

    # Definir acciones permitidas
    action_set = [
        ["right"],         # 0: avanzar
        ["right", "A"],    # 1: salto + avanzar
    ]
    environment = JoypadSpace(environment, action_set)

    # Aplicar wrappers personalizados + estándar
    environment = SkipFrame(environment, skip=4)
    environment = GrayScaleObservation(environment, keep_dim=False)
    environment = ResizeObservation(environment, new_size=84)
    environment = TransformObservation(environment, lambda x: x / 255.0)
    environment = FrameStack(environment, num_stack=4)

    return environment



#Configuración general

env = build_environment()
env.reset()

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = Path("checkpoints") / timestamp
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_file = None

# Crear agente
agent = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=checkpoint_dir,
    checkpoint=checkpoint_file
)

# Logger
logger = MetricLogger(checkpoint_dir)

num_episodes = 40000



#Bucle principal de entrenamiento

for episode in range(num_episodes):
    obs = env.reset()

    while True:
        # Acción del agente
        action = agent.act(obs)

        # Avanzar en el entorno
        next_obs, reward, terminated, info = env.step(action)

        # Guardar transición
        agent.cache(obs, next_obs, action, reward, terminated)

        # Aprendizaje
        q_val, loss_val = agent.learn()

        # Registrar métricas del paso
        logger.log_step(reward, loss_val, q_val)

        # Avanzar estado
        obs = next_obs

        # Fin de episodio
        if terminated or info.get("flag_get"):
            break

    # Registrar el cierre del episodio
    logger.log_episode()

    # Registrar cada 20 episodios
    if episode % 20 == 0:
        logger.record(
            episode=episode,
            epsilon=agent.epsilon,
            step=agent.global_step
        )

