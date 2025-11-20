import os
import json
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from simulation_env import Navigation2DEnv
from datetime import datetime
import argparse
from typing import Union
from stable_baselines3 import SAC


class RewardLoggingCallback(BaseCallback):
    # logging the evaluation results for validation reward graph
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        for info in infos:
            if 'episode' in info:
                ep_reward = info['episode']['r']
                self.logger.record("eval/episode_reward", ep_reward)
        return True


class SyncVecNormalizeCallback(BaseCallback):
    # needed in order to update the evaluation env for periodic evaluation
    def __init__(self, train_env, eval_env):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self):
        self.eval_env.obs_rms = self.train_env.obs_rms
        return True


class RLTrainer:
    """
    Trainer class

    """
    def __init__(self,
                 model_save_path: str = "",
                 total_timesteps: int = 1_000_000,
                 n_envs: int = 8,
                 initial_lr: Union[float, str] = 3e-4,
                 batch_size: int = 512,
                 buffer_size = 100_000,
                 learning_starts = 10_000):
        if model_save_path == "":
            raise ValueError("model_save_path cannot be empty")
        self.model_save_path = os.path.join(model_save_path, 'model')
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.vec_env = None
        self.model = None
        self.buffer_size = buffer_size,
        self.learning_starts = learning_starts,

    def to_dict(self):
        """Serializes the trainer's configuration parameters into a dictionary."""
        return {
            "model_save_path": self.model_save_path,
            "total_timesteps": self.total_timesteps,
            "n_envs": self.n_envs,
            "initial_lr": self.initial_lr,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
        }

    def train_and_save(self):
        """
        Setup, train, save, and cleanup.
        Includes saving the model, environment state
        """
        try:
            print(f"Setting up {self.n_envs} parallel environments...")
            env_fns = [self._make_env_fn for _ in range(self.n_envs)]
            self.vec_env = SubprocVecEnv(env_fns)

            # VecNormalize keeps obs and rewards normalized
            self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True)

            # VecMonitor is required so SB3 can record episode rewards for logging
            self.vec_env = VecMonitor(self.vec_env)
            eval_env = SubprocVecEnv([self._make_env_fn_seed])
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

            eval_env = VecMonitor(eval_env)
            eval_callback = EvalCallback(
                eval_env,
                eval_freq=5_000,  # how often to evaluate
                deterministic=True,
                render=False,
            )

            reward_callback = RewardLoggingCallback()
            sync_callback = SyncVecNormalizeCallback(self.vec_env, eval_env)

            print("Initializing RL model...")

            self.model = SAC(
                "MlpPolicy",
                self.vec_env,
                verbose=1,
                buffer_size=100_000,
                learning_rate=self.initial_lr,
                device='cpu',
                learning_starts=10_000,
                ent_coef='auto',
                batch_size=self.batch_size,
                tensorboard_log=f"{MODEL_PATH}/tensorboard"
            )

            print(f"Starting training {self.model_save_path} for {self.total_timesteps} timesteps...")
            self.model.learn(total_timesteps=self.total_timesteps, callback=[eval_callback, sync_callback, reward_callback])
            print("Training complete.")

            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

            self.model.save(self.model_save_path)
            self.vec_env.save(f"{self.model_save_path}.pkl")

            hparams = self.to_dict()
            json_path = f"{self.model_save_path}.json"
            with open(json_path, 'w') as f:
                json.dump(hparams, f, indent=4)

            print(f"Model and environment saved to {self.model_save_path}, {self.model_save_path}.pkl")

        except Exception as e:
            print(f"An error occurred during training: {e}")
            if self.vec_env:
                self.vec_env.close()
        finally:
            if self.vec_env:
                self.vec_env.close()
                print("Vectorized environment closed.")


    @staticmethod
    def _make_env_fn():
        """Helper method for vectorized environments."""
        return Navigation2DEnv()

    @staticmethod
    def _make_env_fn_seed(seed=42):
        """Helper method for SEEDED environments."""
        return Navigation2DEnv(seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an RL agent with configurable environment settings."
    )

    parser.add_argument(
        '--n_envs',
        type=int,
        default=12,
        help='Number of parallel environments to use for training (default: 12).'
    )

    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=1_000_000,
        help='Number of overall timesteps (default: 1M).'
    )

    args = parser.parse_args()
    MODEL_PATH = f"models/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    trainer = RLTrainer(
        model_save_path=MODEL_PATH,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        initial_lr=3e-4,
        batch_size=256,
        buffer_size=100_000,
        learning_starts=10_000,
    )
    trainer.train_and_save()
