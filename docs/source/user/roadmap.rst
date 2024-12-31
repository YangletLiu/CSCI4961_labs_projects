=======
Roadmap
=======

.. contents:: Table of Contents
   :local:

Introductory modules (Concepts, Gates, Circuits)
================================================
- **Qubits**

  - **Superposition**

    - Start with a :math:`\ket{0}` state.

    - Use Hadamard gate to generate a superposition state.

  - **Bloch Sphere**

    - Identify the named states on the Bloch Sphere:

      - North pole

      - South pole

      - :math:`\ket{+}`

      - :math:`\ket{-}`

      - :math:`\ket{i}`

      - :math:`\ket{-i}`

  - **Measurement**

    - Start with a :math:`\ket{0}` state.

    - Use Hadamard gate to generate a superposition state.

    - Apply measurement on the resulting state (ensure measurement outputs to classical bit).

    - Set up and run the circuit.

  - **Entanglement**

    - Start with a :math:`\ket{00}` state.

    - Use Hadamard gate on first qubit to generate a superposition state.

    - Apply CNOT gate, with first qubit as control and second qubit as target.

    - Measure both qubits.

    - Set up and run the circuit.

- **Gates**

  - **Common Gates**

    - For each common gate, start with a :math:`\ket{0}` state.

    - Apply the common gates:

      - Identity

      - Pauli :math:`X` , :math:`Y` , :math:`Z`

      - Hadamard :math:`H`

      - Phase Gate :math:`S`

      - :math:`T` Gate

    - Measure the resulting states.

  - **Universal Gate Set**

    - Use Universal Gate Set { :math:`\text{CNOT}, H, T` } on a :math:`\ket{0}` state.

    - Construct Pauli :math:`X = HT^{4}H`.

    - Construct Pauli :math:`Y = HT^{4}HT^{2}`.

    - Construct the Toffoli gate (optional).

    - Compare measurements of constructed gate with pre-defined gate.

- **Circuits**

  - **Bell States**

    - Start with a :math:`\ket{00}` state.

    - Construct each Bell State:

      - :math:`\ket{\phi^{+}}` : Use Hadamard gate on first qubit and apply CNOT gate with first qubit as control and second qubit as target.

      - :math:`\ket{\phi^{-}}` : Follow same steps as :math:`\ket{\phi^{+}}`. Apply :math:`Z` gate on first qubit before the CNOT.

      - :math:`\ket{\psi^{+}}` : Follow same steps as :math:`\ket{\phi^{+}}`. Apply :math:`X` gate on second qubit before the CNOT.

      - :math:`\ket{\psi^{-}}` : Follow same steps as :math:`\ket{\psi^{+}}`. Apply a :math:`Z` gate on first qubit and second qubit before the CNOT.

    - Apply measurement on both qubits, set up, and run the circuit.

  - **GHZ State**
    
    - Start with a :math:`\ket{000}` state.

    - Use Hadamard gate on first qubit to generate superposition state.

    - Apply CNOT gate, with first qubit as control and second qubit as target.

    - Apply another CNOT gate, with first qubit as control and third qubit as target.

    - Measure each qubit, set up, and run the circuit.

  - **Error Codes**
    
    - Showcase simple error.

      - Start with a :math:`\ket{00}` state.

      - Apply :math:`X` gate on the second qubit to simulate bit flip error.

    - Construct simple error detection (parity checking).

      - Add a third additional qubit (ancilla).

      - Apply a CNOT gate, with first qubit as control and third qubit as target.

      - Apply another CNOT gate, with second qubit as control and third qubit as target.

      - Measure the ancilla qubit.

  - **Oracles**

    - Construct a simple oracle where  :math:`f(x) = x`.

      - Start with a :math:`\ket{0}` state. This will be the target qubit, :math:`\ket{y}`.

      - Initialize a second qubit, :math:`\ket{x}`, as :math:`\ket{0}` or :math:`\ket{1}` by applying :math:`X` gate.

      - Apply a CNOT gate, with the second qubit as control and first qubit as target. ( :math:`\ket{y} \rightarrow \ket{y ⊕ x}` )

      - Measure target qubit, :math:`\ket{y}`.

    - Construct a simple **phase** oracle where :math:`f(x) = x`.
    
      - Start with a :math:`\ket{0}` state. This will be the target qubit, :math:`\ket{y}`.

      - Apply a :math:`X` gate, followed by a Hadamard gate, to create the :math:`\ket{-}` state.

      - Initialize a second qubit,:math:`\ket{x}`, as :math:`\ket{0}` or :math:`\ket{1}` by applying :math:`X` gate.

      - Apply a CNOT gate, with the second qubit as control and first qubit as target. ( :math:`\ket{y} \rightarrow \ket{y ⊕ x}` )

      - Measure input qubit, :math:`\ket{x}`.

Intermediate modules (Algorithms) (In Progress)
===============================================

- **Deutsch's**

  - Start with a :math:`\ket{0}` state. This will be the target qubit, :math:`\ket{y}`.

  - Initialize a second qubit, :math:`\ket{x}`, as the answer qubit, :math:`\ket{-}`. (Apply :math:`X` and :math:`H` to :math:`\ket{0}`)

  - Apply :math:`H` gate to the target qubit.

  - Construct the oracle :math:`U_f`. (Depends on specific function. Here we construct oracle for :math:`f(x) = x` or '10'.)

    - Apply CNOT gate, with first qubit as control and second qubit as target.

    - Apply :math:`X` gate to answer qubit.

    - Apply :math:`H` gate to target qubit.

  - Measure our target qubit.

- **Bernstein's**

  - Start with a :math:`\ket{000}` state and 3 classical bits.

  - Initialize fourth qubit as :math:`\ket{-}`.

  - Apply :math:`H` on the first, second, and third qubits.

  - Construct oracle :math:`U_f`. (Oracle for '101')

    - Apply CNOT gate, with first qubit as control and fourth qubit as target.

    - Apply CNOT gate, with third qubit as control and fourth qubit as target.

  - Measure first, second, and third qubits.

- **Grover's**

- **Shor's**

- **Variational Quantum Eigensolver (VQE)**

- **Quantum Approximate Optimization Algorithm (QAOA)**

- **Quantum Key Distribution (QKD)**

- **Error Correction**

Advanced modules (Applications) (In Progress)
=============================================
- **Simulation**

- **Chemistry/Drug Discovery**

- **Machine Learning**

- **Cryptography**