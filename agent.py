import torch
import random
import numpy as np
from collections import deque
from pathlib import Path

from neural import MarioNet


class Mario:

    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None, test=False):

        # Parámetros base
        self.test_mode = test
        self.state_shape = state_dim
        self.num_actions = action_dim
        self.save_directory = save_dir

        # Replay buffer
        self.buffer = deque(maxlen=100000)
        self.batch_size = 32

        # Exploración
        self.epsilon = 1.0
        self.epsilon_decay = 0.99999975
        self.epsilon_min = 0.1

        # Descuento
        self.discount = 0.9

        # Control del entrenamiento
        self.global_step = 0
        self.warmup_steps = 10000
        self.update_interval = 3
        self.target_sync_freq = int(1e4)
        self.checkpoint_freq = int(1e5)

        # GPU
        self.use_gpu = torch.cuda.is_available()

        # Modelo Q (online + target)
        self.model = MarioNet(self.state_shape, self.num_actions).float()
        if self.use_gpu:
            self.model = self.model.to("cuda")

        # Cargar checkpoint previo
        if checkpoint:
            self.load(checkpoint)

        # Optimizador y pérdida (nota: optimiza SOLO la red online)
        self.optimizer = torch.optim.Adam(self.model.online.parameters(), lr=2.5e-4)
        self.loss_function = torch.nn.SmoothL1Loss()


    # Acción, Epsilon-greedy
    def act(self, state):
        if (np.random.rand() < self.epsilon) and not self.test_mode:
            action = np.random.randint(self.num_actions)
        else:
            t = torch.FloatTensor(state)
            if self.use_gpu:
                t = t.cuda()

            q_vals = self.model(t.unsqueeze(0), mode="online")
            action = torch.argmax(q_vals, dim=1).item()

        # Decaimiento epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.global_step += 1
        return action


    # Guardar transición

    def cache(self, state, next_state, action, reward, done):

        to_tensor = lambda x: torch.tensor(x).float().cuda() if self.use_gpu else torch.tensor(x).float()

        s = to_tensor(state)
        ns = to_tensor(next_state)

        a = torch.tensor([action]).long()
        r = torch.tensor([reward]).double()
        d = torch.tensor([done]).bool()

        if self.use_gpu:
            a = a.cuda()
            r = r.cuda()
            d = d.cuda()

        self.buffer.append((s, ns, a, r, d))


    # Sample batch

    def recall(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, next_states, actions, rewards, dones = map(torch.stack, zip(*batch))
        return (
            states,
            next_states,
            actions.squeeze(),
            rewards.squeeze(),
            dones.squeeze(),
        )

    
    # TD Estimate
    # Temporal Difference
    def td_estimate(self, states, actions):
        q_vals = self.model(states, mode="online")
        batch_indices = torch.arange(self.batch_size)
        return q_vals[batch_indices, actions]


    # TD Target (Double DQN)

    @torch.no_grad()
    def td_target(self, rewards, next_states, dones):

        q_online_next = self.model(next_states, mode="online")
        best_actions = torch.argmax(q_online_next, dim=1)

        q_target_next = self.model(next_states, mode="target")
        q_best = q_target_next[torch.arange(self.batch_size), best_actions]

        return (rewards + (1 - dones.float()) * self.discount * q_best).float()


    # Actualizar red online

    def update_online(self, pred, target):
        loss = self.loss_function(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    # Sincronizar redes

    def sync_target_network(self):
        self.model.target.load_state_dict(self.model.online.state_dict())


    # Ciclo de aprendizaje completo
    def learn(self):

        # Sync periódica
        if self.global_step % self.target_sync_freq == 0:
            self.sync_target_network()

        # Guardar modelo
        if self.global_step % self.checkpoint_freq == 0:
            self.save()

        # Burn-in
        if self.global_step < self.warmup_steps:
            return None, None

        # Frecuencia de updates
        if self.global_step % self.update_interval != 0:
            return None, None

        # Sample
        states, next_states, actions, rewards, dones = self.recall()

        # TD
        q_pred = self.td_estimate(states, actions)
        q_tgt = self.td_target(rewards, next_states, dones)

        # Update
        loss = self.update_online(q_pred, q_tgt)

        return q_pred.mean().item(), loss

    # Guardado
    def save(self):
        file_path = self.save_directory / f"mario_net_{int(self.global_step // self.checkpoint_freq)}.chkpt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "epsilon": self.epsilon,
            },
            file_path,
        )
        print(f"[SAVE] Modelo guardado en {file_path} (step {self.global_step})")

    
    # Carga
    def load(self, file_path):
        if not file_path.exists():
            raise ValueError(f"Checkpoint no encontrado: {file_path}")

        checkpoint = torch.load(file_path, map_location=("cuda" if self.use_gpu else "cpu"))
        model_weights = checkpoint.get("model")
        epsilon_value = checkpoint.get("epsilon")

        print(f"[LOAD] Cargando modelo desde {file_path} (epsilon={epsilon_value})")

        self.model.load_state_dict(model_weights)

        # Restaurar epsilon si existe
        if epsilon_value is not None:
            self.epsilon = float(epsilon_value)

