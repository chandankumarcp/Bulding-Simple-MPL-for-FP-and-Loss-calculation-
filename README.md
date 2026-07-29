# 🧠 Building a Simple MLP for Forward Pass & Loss Calculation

A clean, from-scratch implementation of a **Multi-Layer Perceptron (MLP)** in Python using **NumPy only** — no deep learning frameworks required. This project demonstrates the core mechanics of a neural network: weight initialization, forward propagation, activation functions, and loss computation.

---

## 📌 Overview

This project builds a simple MLP that supports:

- Arbitrary number of hidden layers and neurons
- **ReLU** activation for hidden layers
- **Sigmoid** activation for the output layer (binary classification)
- Forward pass with layer-wise shape logging
- Two loss functions: **Binary Cross-Entropy (BCE)** and **Mean Squared Error (MSE)**
- He weight initialization (optimal for ReLU networks)
- A network `summary()` showing architecture and parameter counts

---

## 📁 Project Structure

```
Bulding-Simple-MPL-for-FP-and-Loss-calculation-/
│
├── mlp_forward_loss.py   # Main implementation: MLP class + activation & loss functions
└── README.md             # Project documentation
```

---

## 🔧 Requirements

- Python 3.7+
- NumPy

Install NumPy if you haven't already:

```bash
pip install numpy
```

---

## 🚀 Getting Started

Clone the repository and run the demo:

```bash
git clone https://github.com/chandankumarcp/Bulding-Simple-MPL-for-FP-and-Loss-calculation-.git
cd Bulding-Simple-MPL-for-FP-and-Loss-calculation-
python mlp_forward_loss.py
```

---

## 🏗️ Architecture

The MLP is defined by a `layer_sizes` list. For example, `[2, 4, 3, 1]` means:

```
Input Layer  →  2 neurons
Hidden Layer 1  →  4 neurons  (ReLU)
Hidden Layer 2  →  3 neurons  (ReLU)
Output Layer →  1 neuron   (Sigmoid)
```

### Weight Initialization

Weights are initialized using **He initialization**, which is recommended for networks using ReLU activations:

```
W ~ N(0, sqrt(2 / fan_in))
```

---

## ⚙️ Core Components

### Activation Functions

| Function  | Description                                    | Used In         |
|-----------|------------------------------------------------|-----------------|
| `relu`    | max(0, z) — introduces non-linearity           | Hidden layers   |
| `sigmoid` | 1 / (1 + e^-z) — squashes output to (0, 1)    | Output layer    |
| `softmax` | Normalized exponentials for multi-class output | (Available for extension) |

### Loss Functions

| Function                        | Formula                                   | Use Case              |
|---------------------------------|-------------------------------------------|-----------------------|
| `mse_loss`                      | mean((y_true - y_pred)²)                  | Regression            |
| `binary_cross_entropy_loss`     | -mean(y·log(ŷ) + (1-y)·log(1-ŷ))        | Binary classification |
| `categorical_cross_entropy_loss`| -mean(Σ y·log(ŷ))                        | Multi-class (extend)  |

---

## 🧪 Demo Walkthrough

The `__main__` block in `mlp_forward_loss.py` runs a full end-to-end demo:

### 1. Create a Dummy Dataset

```python
X = np.random.randn(5, 2)          # 5 samples, 2 features
y = np.array([[1],[0],[1],[1],[0]]) # Binary labels
```

### 2. Build the MLP

```python
model = MLP(layer_sizes=[2, 4, 3, 1])
model.summary()
```

**Sample output:**
```
========================================
       MLP Architecture Summary
========================================
  [Hidden Layer 1]  W: (2, 4)  b: (1, 4)  Params: 12
  [Hidden Layer 2]  W: (4, 3)  b: (1, 3)  Params: 15
  [Output Layer 3]  W: (3, 1)  b: (1, 1)  Params: 4
----------------------------------------
  Total Trainable Parameters: 31
========================================
```

### 3. Forward Pass

```python
y_pred, cache = model.forward(X)
```

Each layer logs its `Z` (pre-activation) and `A` (post-activation) shapes, making it easy to debug and understand data flow through the network.

### 4. Loss Calculation

```python
bce = model.compute_loss(y, y_pred, loss_type="bce")
mse = model.compute_loss(y, y_pred, loss_type="mse")
```

**Sample output:**
```
  Binary Cross-Entropy Loss : 0.712354
  Mean Squared Error Loss   : 0.251843
```

---

## 📐 Math Behind the Forward Pass

For each layer `i`:

```
Z[i] = A[i-1] · W[i] + b[i]    ← Linear transformation
A[i] = activation(Z[i])         ← Non-linear activation
```

The final output for binary classification:

```
ŷ = sigmoid(Z_output)
```

---

## 🔮 Possible Extensions

- ✅ Add **backpropagation** and **gradient descent** for training
- ✅ Add **dropout** regularization
- ✅ Support **multi-class classification** using softmax + categorical cross-entropy
- ✅ Visualize loss curves over training epochs
- ✅ Add **batch normalization** between layers

---

## 👤 Author

**Chandan Kumar**
- GitHub: [@chandankumarcp](https://github.com/chandankumarcp)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> 💡 *This project is intended for learning purposes — to understand the internals of neural networks before using high-level frameworks like PyTorch or TensorFlow.*
