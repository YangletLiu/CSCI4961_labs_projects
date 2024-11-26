import gym
from gym import spaces
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import HGate, CXGate, TGate, UnitaryGate
from qiskit.quantum_info import Operator

import matplotlib.pyplot as plt
import csv

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

        self.num_qubits = 2
        self.circuit = QuantumCircuit(self.num_qubits)
        self.target_unitary = iswap_matrix  # Change target circuit if needed

        # Define action and observation space
        self.action_space = spaces.Discrete(6)  # Number of possible actions

        # Use a larger observation space to avoid hash collisions
        self.observation_space = spaces.Discrete(5000)  # Adjusted state space size
        
        self.state_to_index = {}
        self.index_to_state = []
        self.possible_actions = [
            [HGate(), [0]],
            [HGate(), [1]],
            [CXGate(), [0, 1]],
            [CXGate(), [1, 0]],
            [TGate(), [0]],
            [TGate(), [1]],
        ]

    def get_state_index(self):
        # Use depth and last gate as state representation
        depth = self.circuit.depth()
        last_gate = self.circuit.data[-1][0].name if self.circuit.data else 'none'
        state = (depth, last_gate)
        if state not in self.state_to_index:
            index = len(self.state_to_index)
            self.state_to_index[state] = index
            self.index_to_state.append(state)
        return self.state_to_index[state]

    def reset(self):
        self.circuit = QuantumCircuit(self.num_qubits)
        self.state_to_index = {}
        self.index_to_state = []
        return self.get_state_index()

    def step(self, action_index):
        action = self.possible_actions[action_index]
        # Execute the action and return next_state, reward, done
        self.circuit.append(action[0], action[1])
        state_index = self.get_state_index()
        reward, done = self._reward(self.target_unitary)
        return state_index, reward, done, {}

    def _reward(self, target_unitary):
        simulator = Aer.get_backend('unitary_simulator')
        transpiled_circuit = transpile(self.circuit, simulator)
        result = simulator.run(transpiled_circuit).result()
        unitary = result.get_unitary(transpiled_circuit)

        unitary_array = np.asarray(unitary)
        target_unitary_array = np.asarray(target_unitary)

        fidelity = np.abs(np.trace(unitary_array.conj().T @ target_unitary_array)) / (2 ** self.num_qubits)
        # reward = fidelity * 100  # Give continuous reward based on fidelity
        reward = 0
        done = False
        if fidelity > 0.99:
            done = True
            reward += 100  # Extra reward for achieving high fidelity
            self.render()
        elif self.circuit.depth() > 6:
            done = True
            reward -= 10  # Penalty for exceeding maximum depth
        return reward, done

    def close(self):
        pass

    def render(self):
        print(self.circuit.draw())

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, state_size, action_size, alpha, gamma, epsilon, decay_rate, epsilon_min):
        # Initialize the agent's parameters
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.decay_rate = decay_rate  # Exploration decay rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        # Initialize the Q-table with small random values
        self.q_table = np.random.rand(state_size, action_size) * 0.01

    def choose_action(self, state_index):
        if np.random.rand() < self.epsilon:
            # Exploration: random action
            action_index = np.random.randint(self.action_size)
        else:
            # Exploitation: choose the best action
            action_index = np.argmax(self.q_table[state_index])
        return action_index

    def update_q_table(self, state_index, action_index, reward, next_state_index):
        # Update the Q-table based on the agent's experience
        best_next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.gamma * self.q_table[next_state_index, best_next_action]
        td_error = td_target - self.q_table[state_index, action_index]
        self.q_table[state_index, action_index] += self.alpha * td_error

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

# Train the agent
def train_agent(agent, environment, episodes, max_steps_per_episode):
    for episode in range(episodes):
        # Reset the environment at the beginning of each episode
        state_index = environment.reset()
        episode_reward = 0
        for step in range(max_steps_per_episode):
            # Choose an action
            action_index = agent.choose_action(state_index)
            
            # Take the action and observe the outcome
            next_state_index, reward, done, _ = environment.step(action_index)
            episode_reward += reward 
            # Update the Q-table
            agent.update_q_table(state_index, action_index, reward, next_state_index)
            
            # Update the state
            state_index = next_state_index
            
            # Check if the episode is done
            if done:
                print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")
                # environment.render()
                break
        else:
            # If the loop wasn't broken, i.e., max_steps_per_episode was reached
            print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f} (Did not reach the goal)")
        
        # Decay the exploration rate
        agent.decay_exploration()
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Epsilon = {agent.epsilon:.4f}")

# Main function
if __name__ == "__main__":
    # Initialize environment and agent
    environment = QuantumEnv()
    agent = QLearningAgent(
        state_size=5000,  # Adjusted state size
        action_size=6,
        alpha=0.5,
        gamma=0.95,
        epsilon=1,
        decay_rate=0.9998,
        epsilon_min=0.1
    )
    # Train the agent
    train_agent(agent, environment, episodes=10000, max_steps_per_episode=20)
