# Reinforcement Learning for Quantum Circuit Design
## Overview 
Quantum computing has the potential to revolutionize computing by solving problems beyond the reach of classical computers. A major hurdle in this field is the circuit design of complex quantum circuits into simpler, implementable gates on quantum hardware. Traditional hand-crafted heuristic methods are often inefficient and inaccurate, making RL an attractive alternative.

This project leverages the power of reinforcement learning to provide an automatic and scalable approach over traditional hand-crafted heuristic methods. We explore two types of Markov Decision Process (MDP) modeling and the application of RL algorithms, such as Q-learning and Deep Q-Networks (DQNs), to the problem of designing quantum circuits. 

## MDP Modeling
### Example
Given two qubits with initial state $\ket{q_1q_0} = \ket{00}$ and a universal gate set $G$ = { $H, T, \text{CNOT}_{01}$ }, the goal is to find a quantum circuit that generates the Bell state $\ket{\Phi^+} = \frac{1}{\sqrt{2}} \left( \ket{00} + \ket{11} \right)$.

|                                  Matrix Representation                                 |                             Reverse Matrix Representation                              |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| ![0](https://github.com/user-attachments/assets/d9491377-228e-46f4-9ad0-48fe751953d2)  | ![1](https://github.com/user-attachments/assets/c20ed46c-3d57-4de0-a14b-570216dda005)  |
|            Actions $\mathcal{A}$ = { $H_0,H_1,T_0,T_1,\text{CNOT}_{01}$ }              |Actions $\mathcal{A}^{-1}$ = { $H_0^{-1},H_1^{-1},T_0^{-1},T_1^{-1},\text{CNOT}_{01}^{-1}$ }|
| State Space $\mathcal{S}$: Represented in the above tree <br> Initial State $U_0 = I_4$, Target State $U = \ket{\Phi^+}$ | State Space $\mathcal{S^{-1}}$: Represented in the above tree <br> Initial State $U_0 = \ket{\Phi^+}$, Target State $U = I_4$  |
| Reward function $\mathcal{R}$: R(s, a) = 100 if we reach the target state $U = \ket{\Phi^+}$, otherwise R(s, a) = 0  |  Reward function $\mathcal{R}$: R(s, a) = 100 if we reach the target state $U = I_4$, otherwise R(s, a) = 0  |

## Q-Learning
The Q-Learning algorithm updates a Q-table during each iteration using the following formula: <br> <br>
$Q^{\text{new}}(S_t, A_t) \leftarrow (1-\alpha) \cdot Q(S_t, A_t) + \alpha \cdot \left(R_{t+1} + \gamma \cdot \max_{a} Q(S_{t+1}, a)\right)$
<br> <br>
where $\alpha$ is the learning rate, <br>
&emsp;&emsp;&nbsp;&nbsp; $Q(S_t, A_t)$ is the current value, <br>
&emsp;&emsp;&nbsp;&nbsp; $R_{t+1}$ is the current reward, <br>
&emsp;&emsp;&nbsp;&nbsp; $\gamma$ is the discount factor, <br>
&emsp;&emsp;&nbsp; $\max_{a} Q(S_{t+1}, a)$ is the estimate of optimal future value

After a sufficient number of iterations, the Q-table converges. Once training is complete, each agent uses its learned Q-table to attempt generating the target circuit in a test environment, selecting the highest Q-value for each state.

<b> Example </b>

![image](https://github.com/user-attachments/assets/473c3806-5202-4d72-9bc9-1213e701b51b)

Following Table 1, at initial state $S_0$, we take action $a = H_0$ and obtain state $S_1$. At state $S_1$, we take  action $a = \text{CNOT}_{01}$ and obtain the target circuit, $\ket{\Phi^+}$.

## Deep Q-Networks (DQNs)
Deep Q-Networks use neural networks to approximate Q-values for each state-action pair. This approach allows DQN to handle environments with larger or continuous state spaces, where maintaining a Q-table is impractical. <br> <br>
For each learning episode, the agent finds the highest Q-value action according to the policy network and chooses the action using Epsilon-Greedy Policy. After each action, the resulting experience—comprising the current state ($s$), the action taken ($a$), the reward received ($R$), the subsequent state ($s'$), and a terminal indicator ($d$)—is stored in the replay buffer.

The DQN algorithm utilizes two neural networks:
- Policy Network: This network is used for action selection during training. It consists of three fully connected layers, each with 128 units. The inputted state passes through these layers, where linear transformations and ReLU activations are applied, resulting in the expected Q-values for each action.
- Target Network: A separate network that stabilizes training by holding fixed parameters for target computation. It is periodically updated with the policy network's weights to mitigate fluctuations and enhance training stability.
