import gym
from gym import spaces
import hashlib
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import HGate, CXGate, SGate, TGate, XGate, YGate, ZGate, TdgGate, UnitaryGate
from qiskit.quantum_info import Operator
import matplotlib.pyplot as plt
import csv

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

bell_state_unitary = Operator(CNOT) @ Operator(np.kron(H, np.eye(2)))
phi_minus =  Operator(np.kron(np.eye(2),Z)) @ Operator(CNOT) @ Operator(np.kron(H, np.eye(2)))
# Define the normalization factor
psi_plus =  Operator(CNOT) @ Operator(np.kron(X, np.eye(2))) @ Operator(np.kron(H, np.eye(2)))
psi_minus = Operator(np.kron(np.eye(2),Z)) @ Operator(CNOT) @ Operator(np.kron(X, np.eye(2))) @ Operator(np.kron(H, np.eye(2)))
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

# Create a 3-qubit GHZ state
ghz_circuit = QuantumCircuit(3)
ghz_circuit.h(0)  # Apply Hadamard gate to the first qubit
ghz_circuit.cx(0, 1)  # Apply CNOT gate between qubit 0 and qubit 1
ghz_circuit.cx(1, 2)  # Apply CNOT gate between qubit 1 and qubit 2

# Convert the circuit to a unitary operator
ghz_circuit = Operator(ghz_circuit)



# def ObjectiveFunction():
#     temp_circuit = QuantumCircuit(3)
#     temp_circuit.h(0)
#     temp_circuit.cx(0,1)
#     temp_circuit.cx(1,2)
#     gate = UnitaryGate(ghz_circuit)
#     new_gate = temp_circuit.to_gate(label="Target")
#     qc_new = QuantumCircuit(3)
#     qc_new.append(gate,[0,1,2])
#     # print(qc_new.draw())
#     return qc_new
def resetCircuit():
    gate = UnitaryGate(iswap_matrix)
    qc_new = QuantumCircuit(2)
    qc_new.append(gate,[0,1])
    # print(qc_new.draw())
    return qc_new
    

class QuantumEnv(gym.Env):
    def __init__(self):
        super(QuantumEnv, self).__init__()

        self.num_qubits = 2
        self.circuit = resetCircuit()
        self.target_unitary = Operator(QuantumCircuit(self.num_qubits))

        # Define action and observation space
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.num_qubits * 4,))

    def _circuit_to_state(self, circuit: QuantumCircuit) -> np.ndarray:
            # Placeholder feature extraction: flatten the unitary matrix of the circuit
            simulator = Aer.get_backend('unitary_simulator')
            result = simulator.run(transpile(circuit, simulator)).result()
            unitary = result.get_unitary(circuit)

            # Flatten the unitary matrix and normalize
            unitary_array = np.asarray(unitary).flatten()
            return unitary_array

    def reset(self):
        self.circuit = resetCircuit()
        return self._circuit_to_state(self.circuit)

    def generate_actions(self,num_qubits):
        gates = [HGate(), TdgGate()]  # Single-qubit gates
        possible_actions = []

        # Single-qubit gate actions
        for gate in gates:
            for qubit in range(num_qubits):
                possible_actions.append([gate, [qubit]])

        # Two-qubit CX (CNOT) gate actions
        for control_qubit in range(num_qubits):
            for target_qubit in range(num_qubits):
                if control_qubit != target_qubit:
                    possible_actions.append([CXGate(), [control_qubit, target_qubit]])
                    
        return possible_actions
    def step(self, action):
        possible_actions = self.generate_actions(self.num_qubits)
        self.circuit.append(possible_actions[action][0], possible_actions[action][1])
        state = self._circuit_to_state(self.circuit)
        reward, done = self._reward(self.target_unitary)
        return state, reward, done

    def _reward(self, target_unitary):
        simulator = Aer.get_backend('unitary_simulator')
        result = simulator.run(transpile(self.circuit, simulator)).result()
        unitary = result.get_unitary(self.circuit)

        unitary_array = np.asarray(unitary)
        target_unitary_array = np.asarray(target_unitary)

        fidelity = np.abs(np.trace(unitary_array.conj().T @ target_unitary_array)) / (2 ** self.num_qubits)

        reward = -5 * self.circuit.size()
        done = False
        if fidelity > 0.99:
            done = True
            reward += 300
            self.render()
        return reward, done
    
    def render(self):
        print(self.circuit.draw())


import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
import random
from collections import namedtuple, deque

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, alpha=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, buffer_size=10000):
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
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
    def choose_actionnoE(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()
    
    def replay(self):
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
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


import numpy as np
import matplotlib.pyplot as plt

def calculate_moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def train_agent(agent, environment, episodes, max_steps_per_episode, save_path='reward_history.csv'):
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
                agent.update_target_net()
                print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
                break
        
        reward_history.append(episode_reward)

        if len(reward_history) >= 200:  # Adjust as needed
            moving_average = np.mean(reward_history[-50:])
            if moving_average > best_average_reward + 0.01 or moving_average < 0:
                best_average_reward = moving_average
                plateau_count = 0
            else:
                plateau_count += 1
            
            if plateau_count >= 100:
                print(f"Training stopped after {episode + 1} episodes: Performance plateau detected.")
                break
        
        if (episode + 1) % 25 == 0:
            agent.update_target_net()
            environment.render()
            print(f"Episode {episode + 1}: Episode Reward = {episode_reward}")
        
        agent.decay_epsilon()

    # Save the reward history to a CSV file
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Reward'])
        for episode, reward in enumerate(reward_history, start=1):
            writer.writerow([episode, reward])
    
    # Plotting the reward history with moving average
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label='Reward per episode', alpha=0.3)
    plt.plot(calculate_moving_average(reward_history, window_size=50), label='Moving Average (50 episodes)', color='red')
    plt.xlabel('Episode', fontweight="bold")
    plt.ylabel('Average Reward', fontweight="bold")
    plt.title('Reward History with Moving Average', fontweight="bold")
    plt.legend()
    plt.grid(True)
    plt.xticks(fontweight = 'bold')
    plt.yticks(fontweight = 'bold')
    plt.savefig('reward_history_moving_average.pdf')  # Save the plot as an image file

    return reward_history  # Return the history if needed for further analysis


def test_agent(agent, environment, episodes, max_steps_per_episode):
    global holder
    for episode in range(episodes):
        state = environment.reset()
        
        for step in range(max_steps_per_episode):
            action = agent.choose_actionnoE(state)
            next_state, reward, done = environment.step(action)
            state = next_state
            
            if done:
                holder+=1
                break
        environment.render()

global holder
holder = 0


if __name__ == "__main__":
    for i in range(100):
        environment = QuantumEnv()
        print("environment setup")
        agent = DQNAgent(state_size=16, action_size=6, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, batch_size=64, buffer_size=10000)
        print("agent set up")
        train_agent(agent, environment, episodes=100, max_steps_per_episode=5)
        print("test Result")
        test_agent(agent, environment, episodes=1, max_steps_per_episode=5)
    print(holder)