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
**Introductory modules (Concepts)**
- Qubits
  - Superposition
    - Start with a $\ket{0}$ state.
    - Use Hadamard gate to generate a superposition state.
  - Bloch Sphere
    - Identify the named states on the Bloch Sphere:
      - North pole
      - South pole
      - $\ket{+}$
      - $\ket{-}$
      - $\ket{i}$
      - $\ket{-i}$
  - Measurement
    - Start with a $\ket{0}$ state.
    - Use Hadamard gate to generate a superposition state.
    - Apply measurement on the resulting state (ensure measurement outputs to classical bit).
    - Set up and run the circuit.
  - Entanglement
    - Start with a $\ket{00}$ state.
    - Use Hadamard gate on first qubit to generate a superposition state.
    - Apply CNOT gate, with first qubit as control and second qubit as target.
    - Measure both qubits.
    - Set up and run the circuit.
- Gates
  - Common Gates
    - For each common gate, start with a $\ket{0}$ state.
    - Apply the common gates:
      - Identity
      - Pauli $X$ , $Y$ , $Z$
      - Hadamard $H$
      - Phase Gate $S$
      - $T$ Gate
    - Measure the resulting states.
  - Universal Gate Set
    - Use Universal Gate Set { $\text{CNOT}, H, T$ } on a $\ket{0}$ state.
    - Construct Pauli $X = HT^{4}H$.
    - Construct Pauli $Y = HT^{4}HT^{2}$.
    - Construct the Toffoli gate (optional).
    - Compare measurements of constructed gate with pre-defined gate.
- Circuits
  - Bell States
    - Start with a $\ket{00}$ state.
    - Construct each Bell State:
      - $\ket{\phi^{+}}$ : Use Hadamard gate on first qubit and apply CNOT gate with first qubit as control and second qubit as target.
      - $\ket{\phi^{-}}$ : Follow same steps as $\ket{\phi^{+}}$. Apply $Z$ gate on first qubit.
      - $\ket{\psi^{+}}$ : Follow same steps as $\ket{\phi^{+}}$. Apply $X$ gate on first qubit.
      - $\ket{\psi^{-}}$ : Follow same steps as $\ket{\psi^{+}}$. Apply $Z$ gate on first qubit.
    - Apply measurement on both qubits, set up, and run the circuit.
  - GHZ State
    - Start with a $\ket{000}$ state.
    - Use Hadamard gate on first qubit to generate superposition state.
    - Apply CNOT gate, with first qubit as control and second qubit as target.
    - Apply another CNOT gate, with first qubit as control and third qubit as target.
    - Measure each qubit, set up, and run the circuit.
  - Error Codes
    - Showcase simple error. 
      - Start with a $\ket{00}$ state.
      - Apply $X$ gate on the second qubit to simulate bit flip error.
    - Construct simple error detection (parity checking).
      - Add a third additional qubit (ancilla).
      - Apply a CNOT gate, with first qubit as control and third qubit as target.
      - Apply another CNOT gate, with second qubit as control and third qubit as target.
      - Measure the ancilla qubit.
  - Oracles
    - Construct a simple oracle where $f(x) = x$.
      - Start with a $\ket{0}$ state. This will be the target qubit, $\ket{y}$.
      - Initialize a second qubit, $\ket{x}$, as $\ket{0}$ or $\ket{1}$ by applying $X$ gate.
      - Apply a CNOT gate, with the second qubit as control and first qubit as target. ( $\ket{y} \rightarrow \ket{y ⊕ x}$ )
      - Measure each qubit.
    
**Intermediate modules (Algorithms run on IBM Quantum System One)**
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
    - Introduce quantum simulation and significance in studying quantum systems.
    - Dive into ```qiskit-aer``` and their quantum circuit simulation to construct example simulations.
    - Present simulation results.
  - Chemistry/Drug discovery
    - Explain role of quantum computing in chemistry and drug discovery (e.g., how algorithms can model molecular interactions).
    - Provide step-by-step guide to simulating molecular systems using VQE or other algorithms.
    - Discuss results, accuracy and potential impacts.
  - Machine Learning
    - Introduce quantum machine learning and potential to enhance classical machine learning methods (e.g., quantum data representation).
    - Use quantum-inspired algorithms or quantum classifiers in various applications.
    - Analyze performance of quantum models compared to classical counterparts.
  - Cryptography
    - Explain potential for quantum computing to enhance cryptography.
    - Provide example use cases of quantum cryptographic protocols (e.g., QKD in network).
    - Analyze security and potential vulnerabilities.

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
