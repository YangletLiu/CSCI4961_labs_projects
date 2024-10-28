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
  - Bernstein's
  - Variational Quantum Eigensolver (VQE)
  - Quantum Approximation Optimization Algorithm (QAOA)
  - Grover's
  - Shor's
  - Quantum Key Distribution (QKD)
  - Error Correction
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
