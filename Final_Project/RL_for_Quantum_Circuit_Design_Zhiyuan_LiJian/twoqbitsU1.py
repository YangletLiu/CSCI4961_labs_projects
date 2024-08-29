import pickle
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Operator
import numpy as np
import hashlib

# 定义Hadamard门
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

# 定义Pauli-X门
X = np.array([[0, 1], [1, 0]])

# 定义Pauli-Z门
Z = np.array([[1, 0], [0, -1]])

# 定义CNOT门
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

# 将门扩展到两个量子比特系统
H1 = np.kron(H, np.eye(2))  # H作用于第一个比特
X2 = np.kron(np.eye(2), X)  # X作用于第二个比特
Z1 = np.kron(Z, np.eye(2))  # Z作用于第一个比特
Z2 = np.kron(np.eye(2), Z)  # Z作用于第二个比特

# 将这些门按照顺序组合
target_gate = CNOT @ Z2 @ Z1 @ X2 @ H1



# 将组合后的矩阵转化为Qiskit的Operator对象
target_unitary = Operator(target_gate)

num_qubits = 2

# Q-learning 参数
num_actions = 12 # 添加了14种操作（包含所有单比特门和CNOT）
num_states = 100  # 假设有100个可能状态
Q = np.zeros((num_states, num_actions))  # Q 表
alpha = 0.5  # 增加学习率
gamma = 0.95  # 增加折扣因子
epsilon = 0.2  # 增加探索率
max_gates = 10  # 最大门数限制

# 奖励函数
def calculate_reward(circuit, target_unitary):
    simulator = Aer.get_backend('unitary_simulator')
    result = simulator.run(transpile(circuit, simulator)).result()
    unitary = result.get_unitary(circuit)

    # 转换为 numpy 数组
    unitary_array = np.asarray(unitary)
    target_unitary_array = np.asarray(target_unitary)

    fidelity = np.abs(np.trace(unitary_array.conj().T @ target_unitary_array)) / (2 ** num_qubits)

    # 计算奖励
    reward = -2 * circuit.size()  # 每增加一个门扣 2 分
    # reward += fidelity * 10  # 根据保真度给予增量奖励
    if fidelity > 0.99:  # 如果达到目标
        reward += 100  # 达到目标加 100 分
    return reward, fidelity

def hash_quantum_circuit(circuit: QuantumCircuit)-> str:
    circuit_str = circuit.draw(output='text').__str__()  # Explicit conversion to string
    
    # Generate a SHA-256 hash of the circuit string
    circuit_hash = hashlib.sha256(circuit_str.encode('utf-8')).hexdigest()
    hash_int = int(circuit_hash, 16)
    return hash_int%100

# 保存结果到文件
def save_results(circuit, Q, episode):
    with open(f'results_episode_{episode}.pkl', 'wb') as f:
        pickle.dump({'circuit': circuit, 'Q': Q}, f)

# Q-learning 训练
def q_learning(num_episodes=100000000): 
    total_rewards = []  # 用于保存每个回合的总奖励

    for episode in range(num_episodes):
        state = 0
        done = False
        circ = QuantumCircuit(num_qubits)
        episode_reward = 0  # 初始化本回合的奖励

        while not done:
            # 选择动作
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q[state])

            # 执行动作
            if action == 0:
                circ.h(0)  # 对 q0 添加 Hadamard 门
            elif action == 1:
                circ.h(1)  # 对 q1 添加 Hadamard 门
            elif action == 2:
                circ.s(0)  # 对 q0 添加 S 门
            elif action == 3:
                circ.s(1)  # 对 q1 添加 S 门
            elif action == 4:
                circ.cx(0, 1)  # 对 (q0, q1) 添加 CNOT 门
            elif action == 5:
                circ.cx(1, 0)  # 对 (q1, q0) 添加 CNOT 门
            elif action == 6:
                circ.t(0)  # 对 q0 添加 T 门
            elif action == 7:
                circ.t(1)  # 对 q1 添加 T 门
            elif action == 8:
                circ.x(0)  # 对 q0 添加 X 门
            elif action == 9:
                circ.x(1)  # 对 q1 添加 X 门
            elif action == 10:
                circ.z(0)  # 对 q0 添加 Z 门
            elif action == 11:
                circ.z(1)  # 对 q1 添加 Z 门

            # 计算奖励和保真度
            reward, fidelity = calculate_reward(circ, target_unitary)
            episode_reward += reward  # 累加本回合的奖励

            # print(f"Reward: {reward}, Fidelity: {fidelity}")

            # 更新 Q 值
            new_state = hash_quantum_circuit(circ)  # 使用模运算确保索引有效
            # print(new_state)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])

            # 检查是否完成：直接根据保真度判断
            if fidelity > 0.99:  # 保真度达到目标值
                done = True

                print("生成的电路:")
                print("Fid: {fidelity}")
                print(circ.draw())
                print(f"Episode {episode + 1}: Total Reward = {episode_reward}")


            # 检查是否超过最大门数
            if circ.size() > max_gates:
                episode_reward -= 100  # 超过最大门数给予负奖励
                break

            state = new_state

        total_rewards.append(episode_reward)  # 记录本回合的奖励

        # 每 100 次尝试保存一次结果
        if (episode + 1) % 100 == 0:
            save_results(circ, Q, episode + 1)
            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
            # print(Q)

    return total_rewards


# 运行 Q-learning
total_rewards = q_learning()

# 打印最终 Q 表
print("最终 Q 表:")
print(Q)

# # 打印每个回合的总奖励
# print("每个回合的总奖励:")
# print(total_rewards)
