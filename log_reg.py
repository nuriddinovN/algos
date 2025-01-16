import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data for Binary Classification
def generate_data():
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Label: 1 if sum of features > 0, else 0
    return X, y

# 2. Sigmoid Function
def sigmoid(z):
    """
    Computes the sigmoid of z.
    Args:
        z: A scalar or numpy array.
    Returns:
        The sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))

# 3. Compute Cost
def compute_cost(X, y, w, b):
    """
    Compute the cost function for logistic regression.
    Args:
        X: ndarray, shape (m, n) - Input data (m samples, n features).
        y: ndarray, shape (m,) - Labels (binary: 0 or 1).
        w: ndarray, shape (n,) - Weight parameters.
        b: float - Bias parameter.
    Returns:
        cost: float - The computed cost.
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    cost = -np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb)) / m
    return cost

# 4. Compute Gradients
def compute_gradient(X, y, w, b):
    """
    Compute the gradient of the cost function with respect to w and b.
    Args:
        X: ndarray, shape (m, n) - Input data.
        y: ndarray, shape (m,) - Labels.
        w: ndarray, shape (n,) - Weight parameters.
        b: float - Bias parameter.
    Returns:
        dj_dw: ndarray, shape (n,) - Gradient with respect to w.
        dj_db: float - Gradient with respect to b.
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    dj_dw = np.dot(X.T, (f_wb - y)) / m
    dj_db = np.sum(f_wb - y) / m
    return dj_dw, dj_db

# 5. Gradient Descent
def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    """
    Perform gradient descent to optimize w and b.
    Args:
        X: ndarray, shape (m, n) - Input data.
        y: ndarray, shape (m,) - Labels.
        w_init: ndarray, shape (n,) - Initial weights.
        b_init: float - Initial bias.
        alpha: float - Learning rate.
        num_iters: int - Number of iterations.
    Returns:
        w: ndarray, shape (n,) - Optimized weights.
        b: float - Optimized bias.
        cost_history: list - Cost at each iteration.
    """
    w = w_init
    b = b_init
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}, w {w}, b {b:.4f}")

    return w, b, cost_history

# 6. Plot Decision Boundary
def plot_decision_boundary(X, y, w, b):
    """
    Plot the data points and the decision boundary.
    Args:
        X: ndarray, shape (m, 2) - Input data.
        y: ndarray, shape (m,) - Labels.
        w: ndarray, shape (2,) - Weight parameters.
        b: float - Bias parameter.
    """
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = -(w[0] * x1 + b) / w[1]

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, label="Data points")
    plt.plot(x1, x2, label="Decision boundary", color="red")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Logistic Regression Decision Boundary")
    plt.show()

# Main Function
if __name__ == "__main__":
    # Generate data
    X, y = generate_data()

    # Initialize parameters
    w_init = np.zeros(X.shape[1])
    b_init = 0.0
    alpha = 0.1
    num_iters = 1000

    # Run gradient descent
    w, b, cost_history = gradient_descent(X, y, w_init, b_init, alpha, num_iters)

    # Print final results
    print(f"\nFinal parameters: w = {w}, b = {b:.4f}")
    print(f"Final cost: {cost_history[-1]:.4f}")

    # Plot cost history
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Reduction over Iterations")
    plt.show()

    # Plot decision boundary
    plot_decision_boundary(X, y, w, b)
