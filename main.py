import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from collections import deque
import random
from environment import BikeEnv
import pygame

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_angle_mean = nn.Linear(64, 1)
        self.fc_angle_std = nn.Linear(64, 1)
        self.fc_speed = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        angle_mean = torch.tanh(self.fc_angle_mean(x))  # Clip angle_mean to [-1, 1]
        angle_std = torch.clamp(torch.exp(self.fc_angle_std(x)), min=1e-6, max=1.0)  # Clip angle_std to [1e-6, 1.0]
        speed_logits = self.fc_speed(x)
        return angle_mean, angle_std, speed_logits

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, clip_epsilon, buffer_size, batch_size):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.optimizer = optim.RMSprop(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        angle_mean, angle_std, speed_logits = self.policy_net(state)

        # Check for NaNs in the network outputs
        if torch.isnan(angle_mean).any() or torch.isnan(angle_std).any() or torch.isnan(speed_logits).any():
            print("Warning: NaNs detected in the network outputs!")

        angle_dist = Normal(angle_mean, angle_std)
        speed_dist = Categorical(logits=speed_logits)

        angle_action = angle_dist.sample()
        speed_action = speed_dist.sample()

        angle_log_prob = angle_dist.log_prob(angle_action)
        speed_log_prob = speed_dist.log_prob(speed_action)

        return (angle_action.item(), speed_action.item()), (angle_log_prob, speed_log_prob)
    
    def update(self, batch):
        states, angle_actions, speed_actions, angle_log_probs, speed_log_probs, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        angle_actions = torch.tensor(angle_actions, dtype=torch.float32)
        speed_actions = torch.tensor(speed_actions, dtype=torch.long)
        angle_log_probs = torch.stack(angle_log_probs)
        speed_log_probs = torch.stack(speed_log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Calculate advantages
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            advantages = rewards + self.gamma * next_values * (1 - dones) - values

        # Calculate value targets
        value_targets = rewards + self.gamma * next_values * (1 - dones)

        # Update policy network
        angle_mean, angle_std, speed_logits = self.policy_net(states)

        angle_dist = Normal(angle_mean, angle_std)
        speed_dist = Categorical(logits=speed_logits)

        angle_ratio = torch.exp(angle_dist.log_prob(angle_actions) - angle_log_probs.detach())
        speed_ratio = torch.exp(speed_dist.log_prob(speed_actions) - speed_log_probs.detach())

        angle_surr1 = angle_ratio * advantages
        angle_surr2 = torch.clamp(angle_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        angle_policy_loss = -torch.min(angle_surr1, angle_surr2).mean()

        speed_surr1 = speed_ratio * advantages
        speed_surr2 = torch.clamp(speed_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        speed_policy_loss = -torch.min(speed_surr1, speed_surr2).mean()

        policy_loss = angle_policy_loss + speed_policy_loss

        # Update value network
        values = self.value_net(states).squeeze()
        value_loss = nn.MSELoss()(values, value_targets.detach())

        # Update networks
        self.optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        self.optimizer.step()

    
    def store_experience(self, state, action, log_prob, reward, next_state, done):
        self.buffer.append((state, action[0], action[1], log_prob[0], log_prob[1], reward, next_state, done))

        # Normalize the reward
        rewards = [exp[5] for exp in self.buffer]
        reward = (reward - np.mean(rewards)) / (np.std(rewards) + 1e-6)
        self.buffer[-1] = (state, action[0], action[1], log_prob[0], log_prob[1], reward, next_state, done)

    def sample_experiences(self):
        if len(self.buffer) >= self.batch_size:
            batch = random.sample(self.buffer, self.batch_size)
            return batch
        else:
            return []

class BikeSimulator:
    def __init__(self, env):
        self.env = env
        self.screen_width = 800
        self.screen_height = 600

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Bike Environment")

    def render(self):
        self.screen.fill((255, 255, 255))  # Clear the screen

        # Draw the bike
        bike_x = int(self.env.position[0] * 100) + 400
        bike_y = int(self.env.position[1] * 100) + 300
        pygame.draw.circle(self.screen, (255, 0, 0), (bike_x, bike_y), 10)

        # Draw the obstacles
        for obstacle in self.env.obstacles:
            obstacle_x = int(obstacle[0] * 100) + 400
            obstacle_y = int(obstacle[1] * 100) + 300
            pygame.draw.circle(self.screen, (0, 0, 255), (obstacle_x, obstacle_y), 10)

        # Draw the goal
        goal_x = int(self.env.goal_position[0] * 100) + 400
        goal_y = int(self.env.goal_position[1] * 100) + 300
        pygame.draw.circle(self.screen, (0, 255, 0), (goal_x, goal_y), 20)  # Increase size to 20

        pygame.display.flip()  # Update the display

    def close(self):
        pygame.quit()

def simulate(env, agent):
    simulator = BikeSimulator(env)
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        simulator.render()

        # Calculate the x and y distances to the target
        target_x, target_y = env.goal_position
        bike_x, bike_y = env.position
        x_distance = target_x - bike_x
        y_distance = target_y - bike_y

        # Provide feedback to the bike about the target coordinates
        print(f"Target Distance: X = {x_distance:.2f}, Y = {y_distance:.2f}")

        # Add a small delay to visualize each step
        pygame.time.delay(100)  # Delay for 100 milliseconds (adjust as needed)

    print(f"Simulation Reward: {total_reward}")

    # Display the reason for the end of the simulation
    if done:
        if env._check_collision():
            print("Simulation ended due to collision with an obstacle.")
        elif np.abs(env.angle) > np.radians(85):
            print("Simulation ended because the bike fell over.")
        else:
            print("Simulation ended because the maximum number of steps was reached.")

    simulator.close()

# Define the obstacle positions and goal position
obstacles = [
    np.array([1.0, 1.0]), np.array([2.0, -1.0]), np.array([-1.0, 2.0]),
    np.array([3.0, 3.0]), np.array([-2.0, -2.0]), np.array([4.0, 0.0]),
    np.array([0.0, 4.0]), np.array([-3.0, 1.0]), np.array([1.0, -3.0]),
    np.array([2.0, 2.0])
]
goal_position = np.array([4.0, 4.0])

# Define the training parameters
state_dim = 22  # Set the state dimension to match the actual size of the state
action_dim = 2
lr = 0.0005
gamma = 0.99
clip_epsilon = 0.3
buffer_size = 10000
batch_size = 32
num_episodes = 1000000
max_steps = 300

# Create the bike environment and agent
env = BikeEnv(obstacles, goal_position)
agent = PPOAgent(state_dim, action_dim, lr, gamma, clip_epsilon, buffer_size, batch_size)

# Train the agent
for episode in range(num_episodes):
    state = env.reset()
    episode_rewards = []

    for step in range(max_steps):
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_experience(state, action, log_prob, reward, next_state, done)
        state = next_state
        episode_rewards.append(reward)

        if done:
            break

    # Sample a batch of experiences and update the agent
    batch = agent.sample_experiences()
    if batch:
        agent.update(batch)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}: Reward = {sum(episode_rewards)}")

    # Run simulation every 2000 episodes
    if (episode + 1) % 2000 == 0:
        print(f"Running simulation after episode {episode + 1}")
        simulate(env, agent)

# Close the environment
env.close()
