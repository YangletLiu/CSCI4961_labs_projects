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
| State Space $\mathcal{S}$: Represented in the above tree <br> Initial State $U_0 = I_4$, Target State $U = \ket{\Phi^+}$ | State Space $\mathcal{S}$: Represented in the above tree <br> Initial State $U_0 = \ket{\Phi^+}$, Target State $U = I_4$  |
| Reward function $\mathcal{R}$: R(s, a) = 100 if we reach the target state $U = \ket{\Phi^+}$, otherwise R(s, a) = 0  |  Reward function $\mathcal{R}$: R(s, a) = 100 if we reach the target state $U = I_4$, otherwise R(s, a) = 0  |

## Q-Learning



## Deep Q-Networks (DQNs)
