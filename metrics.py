import time
import datetime
import numpy as np
import matplotlib.pyplot as plt


class MetricLogger:

    def __init__(self, save_dir):
        # Ruta del archivo de log
        self.log_path = save_dir / "log"

        # Crear encabezado del archivo de registro
        with open(self.log_path, "w") as f:
            header = (
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}"
                f"{'MeanReward':>15}{'MeanLength':>15}"
                f"{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
            f.write(header)

        # Rutas a las gráficas
        self.reward_plot = save_dir / "reward_plot.jpg"
        self.length_plot = save_dir / "length_plot.jpg"
        self.loss_plot = save_dir / "loss_plot.jpg"
        self.q_plot = save_dir / "q_plot.jpg"

        # Historial general
        self.rewards_hist = []
        self.length_hist = []
        self.loss_hist = []
        self.q_hist = []

        # Promedios móviles
        self.avg_reward_hist = []
        self.avg_length_hist = []
        self.avg_loss_hist = []
        self.avg_q_hist = []

        # Estado del episodio actual
        self.reset_episode_stats()

        # Marca de tiempo para delta entre prints
        self.last_time_mark = time.time()

    # -------------------------------------------------

    def reset_episode_stats(self):
        """Reinicia métricas para un nuevo episodio"""
        self.tmp_reward = 0.0
        self.tmp_length = 0
        self.tmp_loss = 0.0
        self.tmp_q = 0.0
        self.tmp_updates = 0

    # -------------------------------------------------

    def log_step(self, reward, loss, q):
        """
        Acumula métricas por step.
        """
        self.tmp_reward += reward
        self.tmp_length += 1

        if loss is not None and loss != 0:
            self.tmp_loss += loss
            self.tmp_q += q
            self.tmp_updates += 1

    # -------------------------------------------------

    def log_episode(self):
        """
        Al final de cada episodio, guarda las métricas en sus historiales.
        """
        self.rewards_hist.append(self.tmp_reward)
        self.length_hist.append(self.tmp_length)

        if self.tmp_updates > 0:
            avg_loss = np.round(self.tmp_loss / self.tmp_updates, 5)
            avg_q = np.round(self.tmp_q / self.tmp_updates, 5)
        else:
            avg_loss = 0
            avg_q = 0

        self.loss_hist.append(avg_loss)
        self.q_hist.append(avg_q)

        self.reset_episode_stats()

    # -------------------------------------------------

    def _compute_moving_average(self, data, window=100):
        """Devuelve promedio móvil de los últimos 100 datos."""
        if len(data) == 0:
            return 0
        return np.round(np.mean(data[-window:]), 3)

    # -------------------------------------------------

    def record(self, episode, epsilon, step):
        """
        Registra progreso y genera gráficos.
        """
        # Calcular promedios móviles
        avg_reward = self._compute_moving_average(self.rewards_hist)
        avg_length = self._compute_moving_average(self.length_hist)
        avg_loss = self._compute_moving_average(self.loss_hist)
        avg_q = self._compute_moving_average(self.q_hist)

        self.avg_reward_hist.append(avg_reward)
        self.avg_length_hist.append(avg_length)
        self.avg_loss_hist.append(avg_loss)
        self.avg_q_hist.append(avg_q)

        # Calcular el tiempo desde el último registro
        prev_time = self.last_time_mark
        self.last_time_mark = time.time()
        delta = np.round(self.last_time_mark - prev_time, 3)

        # Mensaje por consola
        print(
            f"Episode {episode} - Step {step} - Epsilon {epsilon:.3f} - "
            f"Mean Reward {avg_reward} - Mean Length {avg_length} - "
            f"Mean Loss {avg_loss} - Mean Q Value {avg_q} - "
            f"Time Delta {delta} - Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        # Guardar en el archivo de log
        with open(self.log_path, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{avg_reward:15.3f}{avg_length:15.3f}{avg_loss:15.3f}{avg_q:15.3f}"
                f"{delta:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

    
        #Graficar cada métrica
        plots = [
            ("avg_reward_hist", self.reward_plot),
            ("avg_length_hist", self.length_plot),
            ("avg_loss_hist", self.loss_plot),
            ("avg_q_hist", self.q_plot),
        ]

        for attr, path in plots:
            plt.plot(getattr(self, attr))
            plt.savefig(path)
            plt.clf()

