# CSCI4961 Introduction to Quantum Computing: Labs and Projects

## Overview

CSCI 4961 Introduction to Quantum Computing hopes to connect theory with practice, equipping students with the skills necessary to excel in future industry and academic endeavors. As an introductory course, CSCI 4961 covers fundamental quantum computing concepts such as superposition, entanglement, and quantum algorithms. We adopt a hands-on modular approach focused on labs and projects, supplementing theory through engagement in guided practical experiments using Jupyter notebooks and RPI's **IBM Quantum System One**. In this sense, we aim for anyone working on these modules, whether in development or for course labs/projects, to become **quantum-proficient**. Our goal is to expand upon the modules for future iterations of this course and other quantum-related courses at RPI.

## Motivation

The emerging field of quantum computing offers the opportunity for new career paths, research, and collaboration. Contributing to these **open-source** modules facilitates the learning/review of quantum computing fundamentals while also enhancing technical skills suitable for industry. Participants will have the chance to utilize **Qiskit**, one of the highest performing quantum Software Development Kits (SDKs), and experiment with RPI's **IBM Quantum System One**, gaining a competitive edge in the field. Involvement in open-source also enables networking opportunities through collaboration and recognition/credibility as an open-source contributor.

## Final Deliverables

- **Qiskit codes for quantum concepts, algorithms, and applications**
- **Well-documented Jupyter notebook files**
- **Testing on IBM System One on RPI Campus**
- **Reports highlighting performance of the codes**

## Roadmap

### Goals & Milestones
#### <ins>11/01/2024 - 01/01/2025</ins>
**Beginner modules (Concept Visualization and Usage)**
- Qubits
  - Superposition
    - Introduce concept of qubits and ability to exist in multiple states
    - Use Qiskit to visualize superposition states with Hadamard gates to start.
  - Bloch Sphere
    - Explain Bloch Sphere representation of qubits.
    - Use ```qiskit_visualization.plot_bloch_multivector``` to visualize states and other examples on Sphere.
  - Measurement
    - Define measurement and its importance (maybe dive into pulse-level programming).
    - Use ```QuantumCircuit.measure``` to implement measurement and observe collapsed state.
    - Include comparison of measurement outcomes before and after applying gates.
  - Entanglement
    - Introduce concept of entangled states.
    - Construct simple entangled state (e.g., Bell State) using Qiskit.
    - Visualize using ```qiskit.visualization.plot_state_qsphere```.
- Gates
  - Common Gates
    - Describe commonly used gates (e.g., X, Y, Z, H)
    - Provide code examples for creating and applying gates, including visualizations of circuit diagrams.
  - Universal Gate Set
    - Explain what constitutes a Universal Gate Set and its importance.
    - Illustrate how to construct a circuit using a combination of basic gates to create any quantum operation.
- Circuits
  - Bell States
    - Define Bell states and their significance in quantum communication.
    - Use Qiskit to construct and measure Bell states.
    - Demonstrate the properties of Bell states through measurement outcomes.
  - GHZ State
    - Introduce the Greenberger–Horne–Zeilinger (GHZ) state and its applications.
    - Create a GHZ state using Qiskit and visualize the resulting quantum state.
  - Error Codes
    - Explain the importance of quantum error correction.
    - Provide an overview of basic error correction codes (e.g., Shor's code).
    - Include practical examples of implementing a simple error correction scheme using Qiskit.
  - Oracles
    - Define what oracles are in quantum computing.
    - Create a simple oracle function using Qiskit to demonstrate their role in quantum algorithms (e.g., Grover's algorithm).
    - Visualize the circuit involving an oracle.
    
**Intermediate modules (Algorithms)**
  - Deutsch's
    - Introduce Deutsch's algorithm, problem statement (constant vs. balanced function), and its significance in quantum computing.
    - Provide step-by-step code implementation with visualizations.
    - Show how to interpret results and compare classical vs. quantum runtime/solutions.
  - Bernstein's
    - Describe Bernstein-Vazirani algorithm, problem statement (querying hidden string), and classical approach.
    - Provide step-by-step code implementation with visualizations.
    - Show how to interpret results and compare classical vs. quantum runtime/solutions.
  - Variational Quantum Eigensolver (VQE)
    - Introduce VQE and its application in finding the ground state energy of quantum systems.
    - Provide step-by-step code implementation with visualizations.
    - Present the results and discuss convergence and accuracy of the VQE method.
  - Quantum Approximation Optimization Algorithm (QAOA)
    - Explain QAOA and its use in combinatorial optimization problems.
    - Provide code snippets to implement QAOA for a sample optimization problem (e.g., Max Cut)
    - Analyze results and compare with classical optimization techniques.
  - Grover's
    - Introduce Grover's algorithm, problem statement (unstructured search problem), and speedup.
    - Include demonstration of oracle and diffusion operator.
    - Visualize measurement outcomes and probability of finding solution vs. classical methods.
  - Shor's
    - Describe Shor's algorithm, problem statement (factoring large integers), and implication for cryptography.
    - Guide through steps to implement for small to large integers.
    - Discuss Quantum Fourier Transform.
    - Analyze and visualize output.
  - Quantum Key Distribution (QKD)
    - Introduce concept of QKD and importance for secure communication.
    - Provide example of simulating QKD.
    - Analyze security, potential vulnerabilities, and real-world applications.
  - Error Correction
    - Explain necessity of quantum error correction in maintaining qubit fidelity.
    - Introduce common error codes (e.g., CSS, Steane).
    - Provide code examples and visualizations for implementations of encoding and decoding.
    - Analyze reduction in error and discuss challenges in research.
#### <ins>01/01/2025 - 04/01/2025</ins>
**Advanced modules (Applications)**
  - Simulation
  - Chemistry/Drug discovery
  - Machine Learning
  - Cryptography

### Long Term Goals

1. **Promote quantum computing in education and research**
2. **Implementation of additional modules**
3. **Categorize into course curriculum**
4. **Revision of modules based on course feedbacks/evaluations**

## File structure

```
├──Classroom_Sharing_Qiskit_Codes
│   ├── Name1_topic
│   ├── Name2_topic
├── Final Projects
│   ├── Name1_topic
│   ├── Name2_topic
├── IBM Quantum Computing Challenge
│   ├── Name1/lab1, lab2, ...
│   ├── Name2/lab1, lab2, ...
├── Modules
│   ├── Concepts/Module1, Module2, ...
│   ├── Algorithms/Module1, Module2, ...
|   ├── Applications/Module1, Module2, ...
├── Qiskit Global Summer School
│   ├── Name1/lab1, lab2, ...
│   ├── Name2/lab1, lab2, ...
├── Quantum_Circuit_Design
│   ├── method1
│   ├── method2
└── README.md
```  
