import gym
import numpy as np
from skimage import transform
from gym.spaces import Box



class ResizeObservation(gym.ObservationWrapper):
    """
    Redimensiona la observación del entorno al tamaño indicado.
    """
    def __init__(self, env, new_size):
        super().__init__(env)

        # Asegurar que new_size sea una tupla (alto, ancho)
        if isinstance(new_size, int):
            self.target_hw = (new_size, new_size)
        else:
            self.target_hw = tuple(new_size)

        # La observación final mantiene los canales originales
        original_channels = self.observation_space.shape[2:]
        final_shape = self.target_hw + original_channels

        self.observation_space = Box(
            low=0,
            high=255,
            shape=final_shape,
            dtype=np.uint8
        )

    def observation(self, obs):
        # Redimensionar usando skimage.transform
        resized = transform.resize(obs, self.target_hw)

        # Convertir de float [0-1] a uint8 [0-255]
        scaled = (resized * 255).astype(np.uint8)
        return scaled



class SkipFrame(gym.Wrapper):

    def __init__(self, env, skip):
        super().__init__(env)
        self.skip_amount = skip

    def step(self, action):
        accumulated_reward = 0.0
        terminal = False
        info = {}

        for _ in range(self.skip_amount):
            obs, reward, terminal, info = self.env.step(action)
            accumulated_reward += reward

            if terminal:
                break

        return obs, accumulated_reward, terminal, info

