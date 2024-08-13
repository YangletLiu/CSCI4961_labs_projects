import gym
from gym import spaces
import hashlib
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import HGate, CXGate, SGate, TGate, XGate, YGate, ZGate
from qiskit.quantum_info import Operator

# Define target unitary matrices for quantum operations
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])
bell_state_unitary = Operator(CNOT) @ Operator(np.kron(H, np.eye(2)))
cz_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
])
swap_matrix = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])
iswap_matrix = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
])

class QuantumEnv(gym.Env):
    def __init__(self):
        super(QuantumEnv, self).__init__()

        # Initialize quantum circuit environment
        self.num_qubits = 2
        self.circuit = QuantumCircuit(self.num_qubits)
        self.target_unitary = iswap_matrix

        # Define action and observation spaces
        self.action_space = spaces.Discrete(14)  # Number of possible actions
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.num_qubits * 4,))  # Example shape

    def _circuit_to_state(self, circuit: QuantumCircuit) -> np.ndarray:
        # Convert the quantum circuit to its state representation
        simulator = Aer.get_backend('unitary_simulator')
        result = simulator.run(transpile(circuit, simulator)).result()
        unitary = result.get_unitary(circuit)

        # Flatten the unitary matrix and normalize
        unitary_array = np.asarray(unitary).flatten()
        return unitary_array

    def reset(self):
        # Reset the quantum circuit and return initial state
        self.circuit = QuantumCircuit(self.num_qubits)
        return self._circuit_to_state(self.circuit)

    def step(self, action):
        # Apply the chosen action to the quantum circuit
        possible_actions = [
            [HGate(), [0]],
            [HGate(), [1]],
            [CXGate(), [0, 1]],
            [CXGate(), [1, 0]],
            [SGate(), [0]],
            [SGate(), [1]],
            [TGate(), [0]],
            [TGate(), [1]],
            [XGate(), [0]],
            [XGate(), [1]],
            [YGate(), [0]],
            [YGate(), [1]],
            [ZGate(), [0]],
            [ZGate(), [1]],
        ]
        self.circuit.append(possible_actions[action][0], possible_actions[action][1])
        state = self._circuit_to_state(self.circuit)
        reward, done = self._reward(self.target_unitary)
        return state, reward, done

    def _reward(self, target_unitary):
        # Calculate reward based on the fidelity of the target unitary
        simulator = Aer.get_backend('unitary_simulator')
        result = simulator.run(transpile(self.circuit, simulator)).result()
        unitary = result.get_unitary(self.circuit)

        unitary_array = np.asarray(unitary)
        target_unitary_array = np.asarray(target_unitary)

        fidelity = np.abs(np.trace(unitary_array.conj().T @ target_unitary_array)) / (2 ** self.num_qubits)

        reward = -5 * self.circuit.size()
        done = False
        reward += fidelity
        if fidelity > 0.99:
            done = True
            reward += 300
            self.render()
        return reward, done
    
    def render(self):
        # Print the current quantum circuit
        print(self.circuit.draw())


import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Define the neural network architecture for the DQN
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
import random
from collections import namedtuple, deque

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, alpha=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, buffer_size=10000):
        # Initialize the DQN agent
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
    
    def remember(self, state, action, reward, next_state, done):
        # Store experiences in memory
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        # Select an action based on epsilon-greedy strategy
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
    
    def replay(self):
        # Sample experiences and perform a training step
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_net(self):
        # Update the target network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        # Decay epsilon for exploration-exploitation trade-off
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(agent, environment, episodes, max_steps_per_episode):
    # Train the DQN agent
    reward_history = []
    best_average_reward = -float('inf')
    plateau_count = 0

    for episode in range(episodes):
        state = environment.reset()
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = environment.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            episode_reward += reward
            
            if done:
                print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
                break
        
        reward_history.append(episode_reward)

        if len(reward_history) >= 200:  # Adjust as needed
            moving_average = np.mean(reward_history[-50:])
            if moving_average > best_average_reward + 0.01 or moving_average<0:
                best_average_reward = moving_average
                plateau_count = 0
            else:
                plateau_count += 1
            
            if plateau_count >= 10:
                print(f"Training stopped after {episode + 1} episodes: Performance plateau detected.")
                break
        
        if (episode + 1) % 100 == 0:
            agent.update_target_net()
            environment.render()
            print(f"Episode {episode + 1}: Episode Reward = {episode_reward}")
        
        agent.decay_epsilon()

def test_agent(agent, environment, episodes, max_steps_per_episode):
    # Test the trained DQN agent
    for episode in range(episodes):
        state = environment.reset()
        
        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = environment.step(action)
            state = next_state
            
            if done:
                break
        environment.render()


if __name__ == "__main__":
    environment = QuantumEnv()
    agent = DQNAgent(state_size=16, action_size=14, alpha=0.1, gamma=0.95, epsilon=0.9, epsilon_min=0.05, epsilon_decay=0.995, batch_size=64, buffer_size=10000)
    print("Training")
    train_agent(agent, environment, episodes=20000, max_steps_per_episode=20)
    print("Testing")
    test_agent(agent, environment, episodes=10, max_steps_per_episode=5)
