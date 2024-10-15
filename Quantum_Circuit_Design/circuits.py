import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


# Define matrices and operators
swap_matrix = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

# Bell state unitary
bell_state_unitary = Operator(CNOT) @ Operator(np.kron(H, np.eye(2)))
phi_minus = Operator(np.kron(np.eye(2), Z)) @ Operator(CNOT) @ Operator(np.kron(H, np.eye(2)))
psi_plus = Operator(CNOT) @ Operator(np.kron(X, np.eye(2))) @ Operator(np.kron(H, np.eye(2)))
psi_minus = Operator(np.kron(np.eye(2), Z)) @ Operator(CNOT) @ Operator(np.kron(X, np.eye(2))) @ Operator(np.kron(H, np.eye(2)))

# CZ matrix
cz_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
])


# GHZ Circuit (3 qubits)
ghz_circuit = QuantumCircuit(3)
ghz_circuit.h(0)
ghz_circuit.cx(0, 1)
ghz_circuit.cx(1, 2)
ghz_circuit = Operator(ghz_circuit)

# Textbook circuits
# page 200
text_circuit1 = QuantumCircuit(3)
text_circuit1.cx(0,1)
text_circuit1.cx(1,2)
text_circuit1.h(0)
text_circuit1.h(1)
text_circuit1.h(2)
text_circuit1 = Operator(text_circuit1)
# page

test_circuit = QuantumCircuit(3)
test_circuit.h(0)
test_circuit.h(1)
test_circuit.h(2)
test_circuit = Operator(test_circuit)
