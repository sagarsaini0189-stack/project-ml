# STEP 1: Import Deep Learning & Matrix Processing Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from collections import deque
import random

# Fix random seed for reproducible benchmark training
np.random.seed(42)
tf.random.set_seed(42)

# STEP 2: Autonomous Drone Simulation Environment
class DroneNavEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        # Drone starts at the bottom-left coordinate
        self.drone_pos = np.array([0.0, 0.0])
        # Target destination at the top-right coordinate
        self.target_pos = np.array([float(self.grid_size - 1), float(self.grid_size - 1)])
        # Fixed static obstacle coordinate
        self.obstacle_pos = np.array([5.0, 5.0])
        self.steps_taken = 0
        return self._get_state()

    def _get_state(self):
        # State vector: [Drone_X, Drone_Y, Dist_to_Target_X, Dist_to_Target_Y, Dist_to_Obstacle_X, Dist_to_Obstacle_Y]
        diff_target = self.target_pos - self.drone_pos
        diff_obstacle = self.obstacle_pos - self.drone_pos
        state = np.array([
            self.drone_pos[0] / self.grid_size,
            self.drone_pos[1] / self.grid_size,
            diff_target[0] / self.grid_size,
            diff_target[1] / self.grid_size,
            diff_obstacle[0] / self.grid_size,
            diff_obstacle[1] / self.grid_size
        ], dtype=np.float32)
        return state

    def step(self, action):
        # Actions: 0 = Move Up, 1 = Move Down, 2 = Move Left, 3 = Move Right
        step_map = {0: [0, 1], 1: [0, -1], 2: [-1, 0], 3: [1, 0]}
        movement = np.array(step_map[action], dtype=np.float32)
        
        old_dist = np.linalg.norm(self.target_pos - self.drone_pos)
        self.drone_pos = np.clip(self.drone_pos + movement, 0, self.grid_size - 1)
        new_dist = np.linalg.norm(self.target_pos - self.drone_pos)
        
        self.steps_taken += 1
        done = False
        reward = -0.1  # Flight time penalty to encourage shortest path

        # Reward structure: closer to target (+), further away (-)
        if new_dist < old_dist:
            reward += 1.0
        else:
            reward -= 1.0

        # Collision penalty with obstacle
        if np.array_equal(self.drone_pos, self.obstacle_pos):
            reward -= 20.0
            done = True

        # Goal completion reward
        if np.linalg.norm(self.drone_pos - self.target_pos) < 1.0:
            reward += 50.0
            done = True

        # Step cap to prevent infinite wandering
        if self.steps_taken >= 40:
            done = True

        return self._get_state(), reward, done

# STEP 3: Deep Q-Network (DQN) Architecture
def build_dqn_policy(state_dim=6, action_dim=4):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(state_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(action_dim, activation='linear')  # Outputs Q-values for each direction
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# STEP 4: Experience Replay & DQN Training Agent
env = DroneNavEnv()
policy_net = build_dqn_policy()
replay_buffer = deque(maxlen=2000)

gamma = 0.95        # Discount factor
epsilon = 1.0       # Exploration rate
epsilon_decay = 0.98
min_epsilon = 0.05
batch_size = 32
num_episodes = 60

print("--- Initializing Autonomous Drone Flight Training ---")
for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Epsilon-Greedy Action Selection
        if np.random.rand() <= epsilon:
            action = np.random.choice(4)
        else:
            q_values = policy_net.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(q_values[0])

        next_state, reward, done = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Train policy net on historical transitions
        if len(replay_buffer) >= batch_size:
            minibatch = random.sample(replay_buffer, batch_size)
            b_states = np.array([m[0] for m in minibatch])
            b_actions = np.array([m[1] for m in minibatch])
            b_rewards = np.array([m[2] for m in minibatch])
            b_next_states = np.array([m[3] for m in minibatch])
            b_dones = np.array([m[4] for m in minibatch])

            targets = policy_net.predict(b_states, verbose=0)
            next_q = policy_net.predict(b_next_states, verbose=0)

            for i in range(batch_size):
                if b_dones[i]:
                    targets[i, b_actions[i]] = b_rewards[i]
                else:
                    targets[i, b_actions[i]] = b_rewards[i] + gamma * np.max(next_q[i])

            policy_net.fit(b_states, targets, epochs=1, verbose=0)

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    if episode % 10 == 0 or episode == 1:
        print(f"Episode {episode:02d}/{num_episodes} | Total Reward: {total_reward:6.2f} | Epsilon: {epsilon:.3f}")

# STEP 5: Autonomous Autonomous Test Flight (Inference Mode)
print("\n--- Deploying Policy to Test Flight ---")
eval_state = env.reset()
eval_done = False
flight_path = [env.drone_pos.tolist()]

while not eval_done:
    q_vals = policy_net.predict(eval_state.reshape(1, -1), verbose=0)
    best_action = np.argmax(q_vals[0])
    eval_state, _, eval_done = env.step(best_action)
    flight_path.append(env.drone_pos.tolist())

print(f"Waypoints Navigated: {flight_path}")
print(f"Target Reached: {np.linalg.norm(env.drone_pos - env.target_pos) < 1.0}")
