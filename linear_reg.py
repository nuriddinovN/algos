import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data
def generate_data():
    np.random.seed(42)  # For reproducibility
    X = np.random.rand(100, 1) * 10  # 100 data points between 0 and 10
    y = 3 * X + 7 + np.random.randn(100, 1) * 2  # y = 3x + 7 + noise
    return X, y

# 2. Compute the Cost Function
def compute_cost(X, y, w, b):
    """
    Compute the cost (mean squared error) for linear regression.
    
    Args:
        X: ndarray, shape (m, 1) - Input features.
        y: ndarray, shape (m, 1) - Target values.
        w: float - Weight parameter.
        b: float - Bias parameter.
    Returns:
        cost: float - Mean squared error.
    """
    m = X.shape[0]
    cost = np.sum((w * X + b - y) ** 2) / (2 * m)
    return cost

# 3. Compute Gradients
def compute_gradient(X, y, w, b):
    """
    Compute gradients of the cost function w.r.t. w and b.
    
    Args:
        X: ndarray, shape (m, 1) - Input features.
        y: ndarray, shape (m, 1) - Target values.
        w: float - Weight parameter.
        b: float - Bias parameter.
    Returns:
        dj_dw: float - Gradient w.r.t. w.
        dj_db: float - Gradient w.r.t. b.
    """
    m = X.shape[0]
    dj_dw = np.sum((w * X + b - y) * X) / m
    dj_db = np.sum(w * X + b - y) / m
    return dj_dw, dj_db

# 4. Gradient Descent Algorithm
def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    """
    Perform gradient descent to optimize w and b.
    
    Args:
        X: ndarray, shape (m, 1) - Input features.
        y: ndarray, shape (m, 1) - Target values.
        w_init: float - Initial weight.
        b_init: float - Initial bias.
        alpha: float - Learning rate.
        num_iters: int - Number of iterations.
    Returns:
        w: float - Optimized weight.
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
            print(f"Iteration {i}: Cost {cost:.4f}, w {w:.4f}, b {b:.4f}")
    
    return w, b, cost_history

# 5. Plot Results
def plot_results(X, y, w, b):
    plt.scatter(X, y, label="Data", color="blue")
    plt.plot(X, w * X + b, label=f"Prediction: y = {w:.2f}x + {b:.2f}", color="red")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression Fit")
    plt.show()

# Main Function
if __name__ == "__main__":
    # Generate data
    X, y = generate_data()
    
    # Initialize parameters
    w_init = 0.0
    b_init = 0.0
    alpha = 0.01
    num_iters = 1000

    # Run gradient descent
    w, b, cost_history = gradient_descent(X, y, w_init, b_init, alpha, num_iters)
    
    # Print final results
    print(f"\nFinal parameters: w = {w:.4f}, b = {b:.4f}")
    print(f"Final cost: {cost_history[-1]:.4f}")
    
    # Plot cost history
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Reduction over Iterations")
    plt.show()
    
    # Plot regression line
    plot_results(X, y, w, b)
