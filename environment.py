import gym
import numpy as np

class BikeEnv(gym.Env):
    def __init__(self, obstacles, goal_position):
        # Define the action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 5.0]), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(30,))

        # Set up the environment parameters
        self.gravity = 3
        self.mass = 1.0
        self.length = 1.0
        self.dt = 0.1

        # Set up the obstacle positions and goal position
        self.obstacles = obstacles
        self.goal_position = goal_position

        # Set up the initial state
        self.reset()

    def reset(self):
        # Reset the environment to the initial state
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.wheel_angles = np.array([0.0, 0.0])
        self.speed = 0.0  # Initialize speed to zero
        return self._get_obs()

    def step(self, action):
        # Update the state based on the action
        angle_normalized, speed = action
        angle_delta = angle_normalized * 0.9  # Map normalized angle to range [-0.9, 0.9]
        self.angular_velocity += angle_delta

        # Ensure the speed is non-negative (forward direction only)
        self.speed = max(0, speed)

        # Update the position, velocity, and angle based on physics equations
        acceleration = self.gravity * np.sin(self.angle)
        self.velocity = np.array([self.speed * np.cos(self.angle), self.speed * np.sin(self.angle)])
        self.position += self.velocity * self.dt + 0.5 * acceleration * self.dt**2
        self.angle += self.angular_velocity * self.dt
        self.wheel_angles += self.angular_velocity * self.dt

        # Check for collision with obstacles
        if self._check_collision():
            reward = -10.0
            done = True
        else:
            # Calculate the reward based on the distance to the goal
            distance_to_goal = np.linalg.norm(self.goal_position - self.position)
            reward = -distance_to_goal
            done = False

        # Punish the bike for falling down
        if np.abs(self.angle) > np.radians(120):
            reward -= 20.0  # Assign a negative reward for falling down
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Get the relative positions of the nearest obstacles
        nearest_obstacles = self._get_nearest_obstacles(num_obstacles=5)

        # Calculate the x and y distances to the target
        target_x, target_y = self.goal_position
        bike_x, bike_y = self.position
        x_distance = target_x - bike_x
        y_distance = target_y - bike_y

        # Return the current observation, including the goal position, nearest obstacle positions, and target distances
        return np.concatenate([self.position, self.velocity, [self.angle, self.angular_velocity], self.wheel_angles, self.goal_position, nearest_obstacles, [x_distance, y_distance]])
    def _get_nearest_obstacles(self, num_obstacles=5):
        # Calculate the distances to all obstacles
        distances = [np.linalg.norm(self.position - obstacle) for obstacle in self.obstacles]

        # Sort the obstacles based on their distances
        sorted_indices = np.argsort(distances)

        # Select the nearest obstacles
        nearest_indices = sorted_indices[:num_obstacles]

        # Get the relative positions of the nearest obstacles
        nearest_obstacles = [self.obstacles[i] - self.position for i in nearest_indices]

        # Pad the array with zeros if there are fewer than num_obstacles
        nearest_obstacles += [np.zeros(2)] * (num_obstacles - len(nearest_obstacles))

        return np.concatenate(nearest_obstacles)

    def _check_collision(self):
        # Check for collision with obstacles
        for obstacle in self.obstacles:
            if self._is_collision(self.position, obstacle):
                return True
        return False

    def _is_collision(self, position, obstacle):
        # Assuming circular obstacles with radius 0.5
        obstacle_radius = 0.5
        distance = np.linalg.norm(position - obstacle)
        return distance <= obstacle_radius

    def render(self, mode='human'):
        pass  

    def close(self):
        pass  
