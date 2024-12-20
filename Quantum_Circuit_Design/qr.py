import gym
from gym import spaces
import hashlib
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import HGate, CXGate, SGate, TGate, XGate, YGate, ZGate, CRZGate, TdgGate, UnitaryGate
from qiskit.quantum_info import Operator
import matplotlib.pyplot as plt
import csv

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
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

matrix_dict = {}
counter = 0  # 从 0 开始

def matrix_to_hash(matrix):
    matrix_array = np.asarray(matrix)  # 将 `Operator` 转换为 NumPy 数组
    return tuple(tuple(row) for row in matrix_array)

# 检查或插入矩阵并返回对应编号
def get_matrix_id(matrix):
    global counter
    matrix_hash = matrix_to_hash(matrix)
    
    if matrix_hash not in matrix_dict:
        matrix_dict[matrix_hash] = counter
        counter += 1  # 递增编号
    
    return matrix_dict[matrix_hash]

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc = Operator(qc)

def ResetCircuit():
    gate = UnitaryGate(iswap_matrix)
    new = QuantumCircuit(2)
    new.append(gate,[0,1])
    return new



class QuantumEnv(gym.Env):
    def __init__(self):
        super(QuantumEnv, self).__init__()

        self.num_qubits = 2
        self.circuit = ResetCircuit()
        self.target_unitary = Operator(QuantumCircuit(self.num_qubits)) # Change target circuit (bell_state, cz, swap, iswap)

        # Define action and observation space
        self.action_space = spaces.Discrete(6)  # Number of possible actions
        self.observation_space = spaces.Discrete(100)  # Number of possible states (hashes)
        
        self.state_to_index = {}
        self.index_to_state = []

    def _hash_circuit(self, circuit: QuantumCircuit) -> int:
        matrix = Operator(circuit)
        return get_matrix_id(matrix)%100

    def get_state_index(self, state: QuantumCircuit) -> int:
        state_hash = self._hash_circuit(state)
        if state_hash not in self.state_to_index:
            index = len(self.state_to_index)
            self.state_to_index[state_hash] = index
            self.index_to_state.append(state)
        return self.state_to_index[state_hash]

    def get_state_from_index(self, index: int) -> QuantumCircuit:
        if 0 <= index < len(self.index_to_state):
            return self.index_to_state[index]
        return None

    def reset(self):
        self.circuit = ResetCircuit()
        return self.get_state_index(self.circuit)

    def step(self, action, qubits):
        # Execute the action and return next_state, reward, done
        self.circuit.append(action, qubits)
        state_index = self.get_state_index(self.circuit)
        reward, done = self._reward(self.target_unitary)
        return state_index, reward, done

    def render(self):
        print(self.circuit.draw())

    def _reward(self, target_unitary):
        simulator = Aer.get_backend('unitary_simulator')
        result = simulator.run(transpile(self.circuit, simulator)).result()
        unitary = result.get_unitary(self.circuit)

        unitary_array = np.asarray(unitary)
        target_unitary_array = np.asarray(target_unitary)

        fidelity = np.abs(np.trace(unitary_array.conj().T @ target_unitary_array)) / (2 ** self.num_qubits)
        # self.render()
        reward = 0
        done = False
        if fidelity > 0.99:
            done = True
            reward += 100
            self.render()
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
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.epsilon_min = epsilon_min
        # Initialize the Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state_index):
        if np.random.rand() < self.epsilon:
            # Exploration: random action
            action = np.random.randint(self.action_size)
        else:
            # Exploitation: choose the best action
            action = np.argmax(self.q_table[state_index])
        
        possible_actions = [
            [HGate(), [0]],
            [HGate(), [1]],
            [CXGate(), [0, 1]],
            [CXGate(), [1, 0]],
            [TdgGate(), [0]],
            [TdgGate(), [1]],
        ]
        
        return possible_actions[action],action
    
    def choose_actionNoE(self, state_index):
       
        action = np.argmax(self.q_table[state_index])
        
        possible_actions = [
            [HGate(), [0]],
            [HGate(), [1]],
            [CXGate(), [0, 1]],
            [CXGate(), [1, 0]],
            [TdgGate(), [0]],
            [TdgGate(), [1]],
        ]
        
        return possible_actions[action],action
    
    def update_q_table(self, state_index, action, reward, next_state_index):
        # Update the Q-table based on the agent's experience
        self.q_table[state_index, action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state_index]) - self.q_table[state_index, action]
        )
    
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
            action,action_index = agent.choose_action(state_index)
            
            # Take the action and observe the outcome
            next_state_index, reward, done = environment.step(action[0],action[1])
            episode_reward += reward 
            # Update the Q-table
            agent.update_q_table(state_index, action_index, reward, next_state_index)
            
            # Update the state
            state_index = next_state_index
            
            # Check if the episode is done
            if done:
                print("Generated circuit:")
                environment.render()
                print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
                break
            if environment.circuit.size() > 4:
                episode_reward -= 100  # Negative reward for exceeding maximum gates
                break
        
        # Decay the exploration rate
         # Save results every 100 attempts
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
        agent.decay_exploration()

# Test the agent
def test_agent(agent, environment, episodes, max_steps_per_episode):
    for episode in range(episodes):
        # Reset the environment
        environment.reset()
        state_index = environment.reset()

        for step in range(max_steps_per_episode):
            # Choose an action (exploitation only, no exploration)
            # action,action_index = agent.choose_action(state_index)
            
            # Take the action and observe the outcome
            # next_state_index, reward, done = environment.step(action[0],action[1])
            
            action,action_index = agent.choose_actionNoE(state_index)
            
            # Take the action and observe the outcome
            next_state_index, reward, done = environment.step(action[0],action[1])
            
            # Update the state
            state_index = next_state_index
            
            # Check if the episode is done
            if done:
                global holder
                holder+=1
                break
        environment.render()

# Main function

global holder
holder = 0

if __name__ == "__main__":
    # Initialize environment and agent
    
    # print(bell_state_unitary)
    # Train the agent
    # train_agent(agent, environment, episodes=100, max_steps_per_episode=10)
    
    for i in range(20):
        environment = QuantumEnv()
        agent = QLearningAgent(state_size=100, action_size=6, alpha=0.1, gamma=0.95, epsilon=1, decay_rate=0.99, epsilon_min=0.05)
        train_agent(agent, environment, episodes=100, max_steps_per_episode=5)
        print("test Result")
        test_agent(agent, environment, episodes=1, max_steps_per_episode=5)
    print(holder/20)
    
        
        
        
        

        
        
    # print(agent.q_table)
    # print(len(matrix_dict))

    # temp = QuantumCircuit(2)
    # temp.h(1)
    # temp.cx(0,1)
    # temp = Operator(temp)
    # print(temp)
    # temp1 = QuantumCircuit(3)
    # temp1.ccx(0,1,2)
    # print(temp1.draw())
    
    # temp1 = Operator(temp1)
    # print(temp1)
    # print(temp1==temp)