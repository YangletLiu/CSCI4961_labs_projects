import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, TwoLocal
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

import torch
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Dropout2d, MaxPool2d, Flatten, ReLU, Sequential

# Set seed for random generators
algorithm_globals.random_seed = 42
manual_seed(42)

# IBM Quantum Backend Setup
#service = QiskitRuntimeService(channel="ibm_quantum", token="e68f88c24b3afc0137bc62514650ebafb5670c1be2813268e8efded4942faa8b60e6d6bebba2cb521ddd1331d372dbecbbe7dd68ef219aaef5080cb053069838")
#backend = service.least_busy(simulator=False, operational=True)

# Data preparation
batch_size = 1
n_samples = 5

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
X_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

selected_indices = []
for class_idx in range(10):
    class_indices = np.where(np.array(X_train.targets) == class_idx)[0][:n_samples]
    selected_indices.extend(class_indices)
train_subset = Subset(X_train, selected_indices)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

n_samples = 50
X_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

selected_indices_test = []
for class_idx in range(10):
    class_indices_test = np.where(np.array(X_test.targets) == class_idx)[0][:n_samples]
    selected_indices_test.extend(class_indices_test)
test_subset = Subset(X_test, selected_indices_test)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)

# Create QNN
def create_qnn():
    feature_map = ZZFeatureMap(10)
    ansatz = TwoLocal(10, ["rz", "ry", "rz"], "cx", "linear", reps=1)
    print(feature_map.decompose().draw())
    print(ansatz.decompose().draw())
    qc = QuantumCircuit(10)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    return qnn

qnn4 = create_qnn()

class Net(Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(16 * 5 * 5, 64)
        self.fc2 = Linear(64, 10)
        self.qnn = TorchConnector(qnn)  # Apply torch connector
        self.fc3 = Linear(1, 10)  # 10-dimensional output for 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        return x

model4 = Net(qnn4)

optimizer = Adam(model4.parameters(), lr=0.001)
loss_func = CrossEntropyLoss()

# Start training
epochs = 10
loss_list = []
accuracy_list = []
model4.train()

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        output = model4(data)
        loss = loss_func(output, target.long())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / len(train_loader.dataset)
    loss_list.append(avg_loss)
    accuracy_list.append(accuracy)

    print(f"Epoch [{epoch + 1}/{epochs}]\tLoss: {avg_loss:.4f}\tAccuracy: {accuracy:.2f}%")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label="Loss")
plt.title("Hybrid NN Training Convergence")
plt.xlabel("Epochs")
plt.ylabel("CrossEntropy Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracy_list, label="Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.show()

torch.save(model4.state_dict(), "model4.pt")

# Load and evaluate the model
qnn5 = create_qnn()
model5 = Net(qnn5)
model5.load_state_dict(torch.load("model4.pt"))

model5.eval()
with no_grad():
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model5(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = loss_func(output, target.long())
        total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(f"Performance on test data:\n\tLoss: {avg_loss:.4f}\n\tAccuracy: {accuracy:.2f}%")

# Visualize predictions
n_samples_show = 6
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

model5.eval()
with no_grad():
    for count, (data, target) in enumerate(test_loader):
        if count == n_samples_show:
            break
        output = model5(data[0:1])
        pred = output.argmax(dim=1, keepdim=True)

        axes[count].imshow(data[0].numpy().squeeze(), cmap="gray")
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title(f"Predicted {pred.item()}")

plt.show()
