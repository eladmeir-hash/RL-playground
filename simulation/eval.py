import time
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from simulation_env import Navigation2DEnv
from stable_baselines3 import SAC
import os
import argparse
import numpy as np


def evaluate_model(model_path, vecnormalize_path, n_trials=50, render=False):
    """
    Evaluate a trained PPO model on Navigation2DEnv with logging.

    Returns a metrics dictionary and a DataFrame with per-trial logs.
    """

    model = SAC.load(model_path)
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize.load(vecnormalize_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    successes = []
    times_to_success = []
    episode_rewards = []

    trial_logs = []

    # --- 4. Run trials ---
    for trial in range(n_trials):
        obs = eval_env.reset()
        done = False
        t = 0
        success = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

            if render:
                eval_env.envs[0].render()
                time.sleep(0.05)

            t += 1
            ep_reward += reward[0]

            if done[0] and reward[0] > 0:  # success criterion
                success = True

        successes.append(int(success))
        times_to_success.append(t if success else np.nan)
        episode_rewards.append(ep_reward)

        trial_logs.append({
            "trial": trial + 1,
            "success": int(success),
            "steps": t,
            "episode_reward": ep_reward
        })

        print(f"Trial {trial+1}/{n_trials} - Success: {success}, Steps: {t}, Reward: {ep_reward:.2f}")

    # --- 5. Compute aggregate metrics ---
    success_rate = np.mean(successes)
    median_time_to_success = np.nanmedian(times_to_success)
    average_reward = np.mean(episode_rewards)

    return success_rate, median_time_to_success, average_reward




def make_env():
    return Navigation2DEnv(
        max_obstacles=3,
        render_mode="human"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train an RL agent with configurable environment settings."
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='checkpoint path - a directory (str) for model.zip, model.pkl',
        required=True
    )

    parser.add_argument(
        '--render',
        type=str,
        default=None,
        help='Leave blank (None) for statistics, or True for simulation visualization'
    )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=1000,
        help='number of trials for statistical evaluation (default 1000)',
    )

    parser.add_argument(
        '--n_render',
        type=int,
        default=10,
        help='number of simulations for visualization (default 10)',
    )
    args = parser.parse_args()
    checkpoint_directory = args.checkpoint

    pkl_path_ = os.path.join(checkpoint_directory, "model.pkl")
    model_path_ = os.path.join(checkpoint_directory, "model.zip")


    if args.render is not None:
        model_ = SAC.load(model_path_)
        eval_env_ = DummyVecEnv([make_env])
        eval_env_ = VecNormalize.load(pkl_path_, eval_env_)

        eval_env_.training = False
        eval_env_.norm_reward = False
        obs_ = eval_env_.reset()
        n_episodes = args.n_render
        for ep in range(n_episodes):
            eval_env_.envs[0].render()
            done_ = False
            ep_reward_ = 0
            while not done_:
                action_, _ = model_.predict(obs_, deterministic=True)
                obs_, reward_, done_, info_ = eval_env_.step(action_)
                eval_env_.envs[0].render()
                time.sleep(0.05)
                ep_reward_ += reward_[0]
                done_ = done_[0]
            print(f"Episode {ep+1} reward: {ep_reward_:.2f}")
    else:
        all_metrics = {
            "success_rate": [],
            "median_time_to_success": [],
            "average_reward": [],
        }
        for start in range(0, args.n_trials, 20):
            success_rate_, median_time_to_success_, average_reward_ = evaluate_model(
                model_path_,
                pkl_path_,
                n_trials=20,
                render=False
            )
            all_metrics["success_rate"].append(success_rate_)
            all_metrics["median_time_to_success"].append(median_time_to_success_)
            all_metrics["average_reward"].append(average_reward_)

        metrics = {
            "success_rate": np.mean(all_metrics["success_rate"]),
            "median_time_to_success": np.mean(all_metrics["median_time_to_success"]),
            "average_reward": np.mean(all_metrics["average_reward"]),
            "n_trials": args.n_trials,
        }
        print('---------------------------------------------------------------------')
        print(metrics)