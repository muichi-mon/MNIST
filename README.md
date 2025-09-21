# MNIST Classification: ANN vs CNN

## Overview

This project demonstrates **handwritten digit classification** on the **MNIST dataset** using PyTorch. It serves as an introductory project in neural networks, comparing a simple **Artificial Neural Network (ANN)** with a **Convolutional Neural Network (CNN)**.

---

## Features

* Load and preprocess MNIST dataset using `torchvision` transforms.
* Implement a **Multi-Layer Perceptron (MLP)** as an ANN.
* Train and evaluate models on **train/test split**.
* Visualize training progress:

  * Loss curves
  * Accuracy curves
  * Confusion matrix

---

## Installation

```bash
git clone <your-repo-url>
cd <your-repo>
pip install torch torchvision matplotlib numpy pandas scikit-learn
```

---

## Usage

```python
# Load data
train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transforms.ToTensor())

# Create data loaders
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=True)

# Initialize model, loss, optimizer
model = MultiLayerPerceptron()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate
# (loop through epochs, compute train/test accuracy, plot loss/accuracy)
```

---

## Results

* **Training Accuracy:** \~98–99%
* **Test Accuracy:** \~97–98%
* **Confusion Matrix:** Shows per-class prediction performance.
* Visualizations include **loss curves** and **accuracy curves**.

---

## License

This project is open-source and free to use.
