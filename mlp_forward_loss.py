import numpy as np

# ─────────────────────────────────────────────
# Activation Functions
# ─────────────────────────────────────────────

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def softmax(z):
    """Softmax activation for output layer (multi-class)."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────

def mse_loss(y_true, y_pred):
    """Mean Squared Error Loss."""
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy_loss(y_true, y_pred):
    """Binary Cross-Entropy Loss."""
    epsilon = 1e-8  # avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy_loss(y_true, y_pred):
    """Categorical Cross-Entropy Loss (for multi-class)."""
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# ─────────────────────────────────────────────
# MLP Class
# ─────────────────────────────────────────────

class MLP:
    """
    A simple Multi-Layer Perceptron (MLP) supporting:
      - Arbitrary hidden layers
      - ReLU activation for hidden layers
      - Sigmoid activation for binary output
      - Forward pass and loss calculation
    """

    def __init__(self, layer_sizes):
        """
        Initialize the MLP with random weights and biases.

        Args:
            layer_sizes (list): List of integers defining neurons per layer.
                                e.g., [2, 4, 3, 1] means:
                                  - Input layer:   2 neurons
                                  - Hidden layer1: 4 neurons
                                  - Hidden layer2: 3 neurons
                                  - Output layer:  1 neuron
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []

        # He initialization for weights (good for ReLU)
        for i in range(self.num_layers - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """
        Perform the forward pass through the network.

        Args:
            X (np.ndarray): Input data of shape (num_samples, input_features)

        Returns:
            output (np.ndarray): Final output predictions
            cache (list): List of (Z, A) tuples for each layer (useful for backprop later)
        """
        A = X
        cache = []

        # Hidden layers — ReLU activation
        for i in range(self.num_layers - 2):
            Z = A @ self.weights[i] + self.biases[i]  # Linear step
            A = relu(Z)                                # Activation step
            cache.append((Z, A))
            print(f"  Layer {i + 1}: Z shape={Z.shape}, A shape={A.shape}")

        # Output layer — Sigmoid activation (for binary classification)
        Z_out = A @ self.weights[-1] + self.biases[-1]
        A_out = sigmoid(Z_out)
        cache.append((Z_out, A_out))
        print(f"  Output Layer: Z shape={Z_out.shape}, A shape={A_out.shape}")

        return A_out, cache

    def compute_loss(self, y_true, y_pred, loss_type="bce"):
        """
        Compute the loss between true labels and predictions.

        Args:
            y_true (np.ndarray): Ground truth labels
            y_pred (np.ndarray): Predicted values from forward pass
            loss_type (str): 'mse' for Mean Squared Error,
                             'bce' for Binary Cross-Entropy (default)

        Returns:
            loss (float): Computed loss value
        """
        if loss_type == "mse":
            loss = mse_loss(y_true, y_pred)
        elif loss_type == "bce":
            loss = binary_cross_entropy_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss type: '{loss_type}'. Choose 'mse' or 'bce'.")
        return loss

    def summary(self):
        """Print a summary of the network architecture."""
        print("=" * 40)
        print("       MLP Architecture Summary")
        print("=" * 40)
        total_params = 0
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            params = W.size + b.size
            total_params += params
            layer_type = "Hidden" if i < self.num_layers - 2 else "Output"
            print(f"  [{layer_type} Layer {i + 1}]  W: {W.shape}  b: {b.shape}  Params: {params}")
        print("-" * 40)
        print(f"  Total Trainable Parameters: {total_params}")
        print("=" * 40)


# ─────────────────────────────────────────────
# Main — Demo / Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # ── 1. Create dummy dataset ──────────────
    num_samples = 5
    num_features = 2

    X = np.random.randn(num_samples, num_features)   # Shape: (5, 2)
    y = np.array([[1], [0], [1], [1], [0]])           # Binary labels: Shape (5, 1)

    print("Input X:")
    print(X)
    print("\nTrue Labels y:")
    print(y)

    # ── 2. Build the MLP ─────────────────────
    # Architecture: 2 inputs -> 4 hidden -> 3 hidden -> 1 output
    layer_sizes = [2, 4, 3, 1]
    model = MLP(layer_sizes)
    model.summary()

    # ── 3. Forward Pass ──────────────────────
    print("\n[Forward Pass]")
    y_pred, cache = model.forward(X)

    print("\nPredictions (y_pred):")
    print(y_pred)

    # ── 4. Loss Calculation ──────────────────
    print("\n[Loss Calculation]")
    bce = model.compute_loss(y, y_pred, loss_type="bce")
    mse = model.compute_loss(y, y_pred, loss_type="mse")

    print(f"  Binary Cross-Entropy Loss : {bce:.6f}")
    print(f"  Mean Squared Error Loss   : {mse:.6f}")
