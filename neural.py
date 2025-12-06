import copy
from torch import nn


class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        channels, height, width = input_dim

        # Validación de tamaño esperado
        if height != 84:
            raise ValueError(f"Expected height 84, received {height}")
        if width != 84:
            raise ValueError(f"Expected width 84, received {width}")

        # Red principal (Q-online)
        self.online = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        # Red objetivo (Q-target)
        self.target = copy.deepcopy(self.online)

        # Congelar parámetros de la red target
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, mode):
        if mode == "online":
            return self.online(x)
        elif mode == "target":
            return self.target(x)
        else:
            raise ValueError("mode must be 'online' or 'target'")

