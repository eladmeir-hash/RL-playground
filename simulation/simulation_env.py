import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from datetime import datetime
from math import pi, atan2

matplotlib.use("TkAgg")


class Navigation2DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, max_obstacles=3, save_scenes=False, seed=None):
        super().__init__()
        self.np_random = None

        # --- Environment parameters ---
        self.bounds = (-5.0, 5.0)
        self.dt = 0.25
        self.tau = 0.25
        self.v_max = 1.0
        self.goal_radius = 0.25
        self.alpha = np.exp(-self.dt / self.tau)

        # --- Obstacle parameters ---
        self.max_obstacles = max_obstacles
        self.min_obstacle_radius = 0.3
        self.max_obstacle_radius = 1.0
        self.max_obstacle_area_fraction = 0.2
        self.obstacles = np.zeros((self.max_obstacles, 3), dtype=np.float32)  # x, y, r

        # --- Observation: agent + goal + relative obstacles ---
        # Assuming self.bounds[1] is 5 (max X/Y position is 5, so space is -5 to 5)
        # Max distance = sqrt(10^2 + 10^2) = 14.14
        MAX_DISTANCE = 15.0
        MAX_ANGLE = float(pi)

        # The structure of the observation vector:
        # [Ax, Ay, Avx, Avy, R_goal, Theta_goal] + [R_obs, Theta_obs, R_obs_radius] * N
        # Dimensions: 4 (agent state) + 2 (goal polar) + 3 * N (obstacles polar)

        # Initial high bounds array (positive values)
        obs_high = np.array(
            # Agent State (x, y, vx, vy)
            [5, 5, self.v_max, self.v_max] +
            # Goal Polar (R_goal, Theta_goal)
            [MAX_DISTANCE, MAX_ANGLE] +
            # Obstacles Polar (R_obs, Theta_obs, R_obs_radius) * N
            [MAX_DISTANCE, MAX_ANGLE, self.max_obstacle_radius] * self.max_obstacles,
            dtype=np.float32
        )

        # Initial low bounds array (negative values)
        # The bounds for angles (Theta_goal, Theta_obs) are correctly set to [-pi, pi]
        obs_low = -obs_high
        # Goal Distance (R_goal): Index 4 (must be >= 0)
        obs_low[4] = 0.0
        # Obstacle Distances (R_obs): Indices 6, 9, 12... (must be >= 0)
        obs_low[6::3] = 0.0
        # Obstacle Radii (R_obs_radius): Indices 8, 11, 14... (must be >= 0)
        obs_low[8::3] = 0.0

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.v_max, -self.v_max], dtype=np.float32),
            high=np.array([self.v_max, self.v_max], dtype=np.float32),
            dtype=np.float32
        )

        self.agent_pos = None
        self.agent_vel = None
        self.goal_pos = None
        self.t = 0
        self.render_mode = render_mode

        self.k_p = 2.0
        self.prox_scale = 8.0
        self.sigma = 1.0
        self.max_episode_steps = 80

        self.fig = None
        self.ax = None
        self.save_scenes = save_scenes
        self.fig_filename = ""
        self._is_drawn = False

        # setting the seed for evaluation
        if seed is not None:
            self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        rng = np.random.default_rng(seed)
        if self.save_scenes:
            self.fig_filename = ('report/scenes/' +
                                 f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}") + '.png'
            self._is_drawn = False
        # Initialize Obstacles (Centers and Radii)
        self.obstacles[:] = 0.0
        arena_area = (self.bounds[1] - self.bounds[0]) ** 2
        max_area = arena_area * self.max_obstacle_area_fraction
        total_area = 0.0
        num_obstacles = rng.integers(0, self.max_obstacles + 1)

        obstacle_list = []

        for _ in range(num_obstacles):
            radius = rng.uniform(self.min_obstacle_radius, self.max_obstacle_radius)
            area = np.pi * radius ** 2
            if total_area + area > max_area:
                break

            # Loop to find a center that doesn't overlap with existing obstacles
            num_of_trials = 0
            while True:
                num_of_trials += 1
                if num_of_trials >= 20:
                    break
                center = rng.uniform(self.bounds[0] + radius, self.bounds[1] - radius, size=(2,))

                # Check for overlap with existing obstacles
                overlap = False
                for existing_center, existing_radius in obstacle_list:
                    min_dist = radius + existing_radius
                    if self.dist_sq(center, existing_center) < min_dist ** 2:
                        overlap = True
                        break

                if not overlap:
                    obstacle_list.append((center, radius))
                    self.obstacles[len(obstacle_list) - 1] = np.array([center[0], center[1], radius], dtype=np.float32)
                    total_area += area
                    break


        # Initialize Agent Position
        min_agent_dist_to_wall = 0.5

        while True:
            self.agent_pos = rng.uniform(self.bounds[0] + min_agent_dist_to_wall,
                                         self.bounds[1] - min_agent_dist_to_wall, size=(2,))

            # Check for overlap with obstacles
            collision = False
            for center, radius in obstacle_list:
                if self.dist_sq(self.agent_pos,
                           center) < radius ** 2:  # Agent is a point, so we just check if it's inside the radius
                    collision = True
                    break

            if not collision:
                break

        self.agent_vel = np.zeros(2, dtype=np.float32)

        # Initialize Goal Position
        min_separation_radius = self.goal_radius

        while True:
            self.goal_pos = rng.uniform(self.bounds[0] + min_separation_radius,
                                        self.bounds[1] - min_separation_radius, size=(2,))

            # Check for overlap with Agent
            if self.dist_sq(self.goal_pos, self.agent_pos) < min_separation_radius ** 2:
                continue

            # Check for overlap with Obstacles
            collision = False
            for center, radius in obstacle_list:
                # Check if the goal position is too close to the obstacle's surface
                if self.dist_sq(self.goal_pos, center) < (radius + min_separation_radius) ** 2:
                    collision = True
                    break

            if not collision:
                break

        self.t = 0
        return self._get_obs(), {}


    def step(self, action):
        # save old distance for progress reward
        old_distance = np.linalg.norm(self.agent_pos - self.goal_pos)

        # Clip ||action|| <= v_max
        u = np.clip(action, -self.v_max, self.v_max)

        # Update velocity & position
        self.agent_vel = self.alpha * self.agent_vel + (1 - self.alpha) * u
        self.agent_pos += self.agent_vel * self.dt

        # save new distance for progress reward
        new_distance = np.linalg.norm(self.agent_pos - self.goal_pos)

        self.t += 1

        terminated = False
        reward = -1.0  # time penalty

        # combined exponential + linear proximity reward
        p_old = self.prox_scale * np.exp(-old_distance / self.sigma)
        p_new = self.prox_scale * np.exp(-new_distance / self.sigma)
        reward_proximity = (p_new - p_old)
        reward += reward_proximity

        reward += self.k_p * (old_distance - new_distance)
        # Check wall collision
        if not (self.bounds[0] <= self.agent_pos[0] <= self.bounds[1]) or \
           not (self.bounds[0] <= self.agent_pos[1] <= self.bounds[1]):
            terminated = True
            reward = -100.0
            if self.save_scenes:
                name, ext = os.path.splitext(self.fig_filename)
                crashed_fig_filename = f"{name}_crashed{ext}"
                self.render()
                self.fig.savefig(crashed_fig_filename, dpi=300)


        # Check obstacle collisions
        for ox, oy, r in self.obstacles:
            if r > 0 and np.linalg.norm(self.agent_pos - np.array([ox, oy])) <= r:
                terminated = True
                reward = -100.0
                if self.save_scenes:
                    name, ext = os.path.splitext(self.fig_filename)
                    crashed_fig_filename = f"{name}_crashed{ext}"
                    self.render()
                    self.fig.savefig(crashed_fig_filename, dpi=300)
                break

        # Check goal
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        if dist_to_goal <= self.goal_radius:
            terminated = True
            reward = 500.0
            if self.save_scenes and os.path.exists(self.fig_filename):
                os.remove(self.fig_filename)

        obs = self._get_obs()
        truncated = False
        # Check Maximum Steps
        if self.t >= self.max_episode_steps:
            reward -= 50.0
            truncated = True
            if self.save_scenes:
                name, ext = os.path.splitext(self.fig_filename)
                crashed_fig_filename = f"{name}_crashed{ext}"
                self.render()
                self.fig.savefig(crashed_fig_filename, dpi=300)
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Agent Position (x, y) and Velocity (vx, vy) remain in Cartesian
        obs = [
            self.agent_pos[0],
            self.agent_pos[1],
            self.agent_vel[0],
            self.agent_vel[1]
        ]

        # goal relative position (dx, dy)
        dx_goal = self.goal_pos[0] - self.agent_pos[0]
        dy_goal = self.goal_pos[1] - self.agent_pos[1]

        # polar relative agent to goal
        r_goal = np.linalg.norm(self.goal_pos - self.agent_pos)
        theta_goal = atan2(dy_goal, dx_goal)

        obs.extend([r_goal, theta_goal])

        # obstacles relative to agent
        for ox, oy, r_obs in self.obstacles:
            dx_obs = ox - self.agent_pos[0]
            dy_obs = oy - self.agent_pos[1]
            r_obs_center = np.linalg.norm(np.array([dx_obs, dy_obs]))
            theta_obs = atan2(dy_obs, dx_obs)
            obs.extend([r_obs_center, theta_obs, r_obs])

        return np.array(obs, dtype=np.float32)

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()
            self.ax.set_xlim(self.bounds)
            self.ax.set_ylim(self.bounds)
            self.ax.set_aspect('equal')

        self.ax.clear()
        self.ax.set_xlim(self.bounds)
        self.ax.set_ylim(self.bounds)
        self.ax.set_aspect('equal')

        # Draw goal
        circle = plt.Circle(self.goal_pos, self.goal_radius, color='green', alpha=0.3)
        self.ax.add_patch(circle)

        # Draw obstacles
        for ox, oy, r in self.obstacles:
            if r > 0:
                circle = plt.Circle((ox, oy), r, color='red', alpha=0.5)
                self.ax.add_patch(circle)

        # Draw agent
        self.ax.plot(self.agent_pos[0], self.agent_pos[1], 'ro')

        plt.draw()
        if self.save_scenes and not self._is_drawn:
            self.fig.savefig(self.fig_filename, dpi=300)
            self._is_drawn = True
        plt.pause(0.001)


    def close(self):
        if hasattr(self, "fig"):
            # Close the figure
            plt.close(self.fig)
            # Delete the references to the figure and axes
            del self.fig
            del self.ax

    @staticmethod
    def dist_sq(pos1, pos2):
        """Helper function for distance check (squared distance for efficiency)"""
        return np.sum((pos1 - pos2) ** 2)


# ------------------ Debug / Run ------------------
if __name__ == "__main__":
    plt.ion()
    env = Navigation2DEnv(render_mode="human")
    obs_, _ = env.reset()

    for step in range(200):
        action_ = env.action_space.sample()
        obs_, reward_, terminated_, truncated_, info = env.step(action_)
        env.render()
        time.sleep(0.05)
        if terminated_ or truncated_:
            print(f"Episode finished at step {step}, reward {reward_}")
            break

    env.close()
    plt.ioff()
    plt.show()
