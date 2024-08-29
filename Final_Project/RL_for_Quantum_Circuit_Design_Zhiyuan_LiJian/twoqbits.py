import pickle
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Operator
import numpy as np
import hashlib


# Create a quantum circuit with two qubits
qc = QuantumCircuit(2)

# Apply Hadamard gate to the first qubit
qc.h(0)

# Apply CNOT gate with the first qubit as control and the second qubit as target
qc.cx(0, 1)

print(qc.draw())

# Convert to Qiskit Operator object
target_unitary = Operator(qc)
num_qubits = 2

# Q-learning parameters
num_actions = 5  # Added 14 types of operations (including all single-qubit gates and CNOT)
num_states = 31  # Assume 31 possible states
Q = np.zeros((num_states, num_actions))  # Q-table
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
max_gates = 2  # Maximum number of gates allowed

action_sequence_to_state = {
    (): 0,
    ('H0',): 1,
    ('H1',): 2,
    ('T0',): 3,
    ('T1',): 4,
    ('CNOT01',): 5,
    ('H0', 'H0'): 6,
    ('H0', 'H1'): 7,
    ('H0', 'T0'): 8,
    ('H0', 'T1'): 9,
    ('H0', 'CNOT01'): 10,
    ('H1', 'H0'): 11,
    ('H1', 'H1'): 12,
    ('H1', 'T0'): 13,
    ('H1', 'T1'): 14,
    ('H1', 'CNOT01'): 15,
    ('T0', 'H0'): 16,
    ('T0', 'H1'): 17,
    ('T0', 'T0'): 18,
    ('T0', 'T1'): 19,
    ('T0', 'CNOT01'): 20,
    ('T1', 'H0'): 21,
    ('T1', 'H1'): 22,
    ('T1', 'T0'): 23,
    ('T1', 'T1'): 24,
    ('T1', 'CNOT01'): 25,
    ('CNOT01', 'H0'): 26,
    ('CNOT01', 'H1'): 27,
    ('CNOT01', 'T0'): 28,
    ('CNOT01', 'T1'): 29,
    ('CNOT01', 'CNOT01'): 30,
}

# Reward function
def calculate_reward(circuit, target_unitary):
    simulator = Aer.get_backend('unitary_simulator')
    result = simulator.run(transpile(circuit, simulator)).result()
    unitary = result.get_unitary(circuit)

    # Convert to numpy array
    unitary_array = np.asarray(unitary)
    target_unitary_array = np.asarray(target_unitary)

    fidelity = np.abs(np.trace(unitary_array.conj().T @ target_unitary_array)) / (2 ** num_qubits)

    # Calculate reward
    reward = 0
    if fidelity > 0.99:  # If the target is achieved
        reward += 100  # Add 100 points for achieving the target
    return reward, fidelity

# Save results to file
def save_results(circuit, Q, episode):
    with open(f'results_episode_{episode}.pkl', 'wb') as f:
        pickle.dump({'circuit': circuit, 'Q': Q}, f)

def q_learning(num_episodes=300):
    total_rewards = []
    state_to_circuit = {0: QuantumCircuit(num_qubits)}  # Initialize circuit for state 0
    action_history = []  # Save action history for each episode

    for episode in range(num_episodes):
        state = 0
        done = False
        circ = QuantumCircuit(num_qubits)
        episode_reward = 0
        action_history.clear()  # Clear action history

        while not done:
            # Choose an action
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q[state])

            # Execute the action and record it
            if action == 0:
                circ.h(0)
                action_history.append('H0')
            elif action == 1:
                circ.h(1)
                action_history.append('H1')
            elif action == 2:
                circ.t(0)
                action_history.append('T0')
            elif action == 3:
                circ.t(1)
                action_history.append('T1')
            elif action == 4:
                circ.cx(0, 1)
                action_history.append('CNOT01')

            # Check if the circuit is empty; if so, set it to the initial state
            if circ.size() == 0:
                new_state = 0
            else:
                # Find the new state based on the action history
                action_tuple = tuple(action_history)
                if action_tuple in action_sequence_to_state:
                    new_state = action_sequence_to_state[action_tuple]
                else:
                    new_state = state  # If no corresponding state is found, keep the current state

            reward, fidelity = calculate_reward(circ, target_unitary)
            episode_reward += reward

            # Update Q-table
            Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])
            state = new_state

            # Save the circuit corresponding to the state
            state_to_circuit[state] = circ.copy()

            if fidelity > 0.99 or circ.size() == max_gates:
                done = True

        total_rewards.append(episode_reward)
        if (episode + 1) % 100 == 0:
            save_results(circ, Q, episode + 1)
            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    return total_rewards, state_to_circuit


def test_trained_q_table(q_table, state_to_circuit, target_unitary):
    simulator = Aer.get_backend('unitary_simulator')
    
    # 从最初状态开始测试
    state = 0
    circuit = QuantumCircuit(num_qubits)
    actions_taken = []

    # 假设我们要构建的电路最多包含两个门
    for _ in range(2):
        action = np.argmax(q_table[state])
        if action == 0:
            circuit.h(0)
            actions_taken.append('H0')
        elif action == 1:
            circuit.h(1)
            actions_taken.append('H1')
        elif action == 2:
            circuit.t(0)
            actions_taken.append('T0')
        elif action == 3:
            circuit.t(1)
            actions_taken.append('T1')
        elif action == 4:
            circuit.cx(0, 1)
            actions_taken.append('CNOT01')

        # 更新状态
        action_tuple = tuple(actions_taken)
        if action_tuple in action_sequence_to_state:
            state = action_sequence_to_state[action_tuple]
        else:
            state = 0  # 如果没有对应的状态，返回初始状态

    # 模拟电路并计算保真度
    result = simulator.run(transpile(circuit, simulator)).result()
    unitary = result.get_unitary(circuit)
    unitary_array = np.asarray(unitary)
    target_unitary_array = np.asarray(target_unitary)
    fidelity = np.abs(np.trace(unitary_array.conj().T @ target_unitary_array)) / (2 ** num_qubits)

    # 输出测试结果
    print("---------------------")
    print("Tested Circuit:")
    print(circuit.draw())
    print("Action Sequence Taken:", actions_taken)
    print("Fidelity with Target:", fidelity)


# Run Q-learning
total_rewards, state_to_circuit = q_learning()

# Print the final Q-table and corresponding states
print("Final Q-Table and Corresponding States:")
for action_tuple, state in sorted(action_sequence_to_state.items(), key=lambda x: x[1]):
    if state in state_to_circuit:
        circuit = state_to_circuit[state]
        print(f"State {state} (Action Sequence: {action_tuple}):")
        print(circuit.draw())
        print("Q values:", Q[state])
        print("-" * 50)
    elif state == 0:  # Ensure state 0 is also printed
        print(f"State 0 (Initial empty state):")
        print(state_to_circuit[0].draw())
        print("Q values:", Q[0])
        print("-" * 50)

# Print Q-table
print("Final Q-Table:")
print(Q)


# 使用你训练完成的Q-table和电路状态来测试
test_trained_q_table(Q, state_to_circuit, target_unitary)