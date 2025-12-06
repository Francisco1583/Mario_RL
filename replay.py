import datetime
from pathlib import Path
import random

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from agent import Mario
from metrics import MetricLogger
from wrappers import ResizeObservation, SkipFrame



def create_env():
    base_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

    # Acciones permitidas
    actions = [
        ["right"],
        ["right", "A"],
    ]
    base_env = JoypadSpace(base_env, actions)

    # Remover límite de frames
    base_env._max_episode_steps = float("inf")
    # Wrappers
    base_env = SkipFrame(base_env, skip=4)
    base_env = GrayScaleObservation(base_env, keep_dim=False)
    base_env = ResizeObservation(base_env, new_size=84)
    base_env = TransformObservation(base_env, lambda x: x / 255.0)
    base_env = FrameStack(base_env, num_stack=4)

    return base_env

env = create_env()
env.reset()

run_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = Path("trained_mario.chkpt")

agent = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=run_dir,
    checkpoint=checkpoint_path,
    test=True                
)

logger = MetricLogger(run_dir)

episodes = 100



# Bucle de reproducción de episodios

for ep in range(episodes):
    obs = env.reset()

    while True:
        env.render()

        # Acción del modelo cargado
        act = agent.act(obs)

        new_obs, reward, terminal, info = env.step(act)

        # Guardar transición, aunque no entrenamos
        agent.cache(obs, new_obs, act, reward, terminal)

        # Registrar solo recompensas (sin pérdida ni Q)
        logger.log_step(reward, None, None)

        obs = new_obs

        if terminal:
            print("Fin del episodio. info =", info)

        if terminal or info.get("flag_get"):
            break

    logger.log_episode()

    if ep % 20 == 0:
        logger.record(
            episode=ep,
            epsilon=agent.epsilon,
            step=agent.global_step,
        )

