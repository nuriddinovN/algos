import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function with regularization
def compute_cost_reg(X, y, w, b, lambda_):
    """
    Computes the regularized cost for logistic regression.
    """
    m = len(y)
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
    # Regularization term (excluding bias)
    reg_term = (lambda_ / (2 * m)) * np.sum(np.square(w))
    
    # Logistic regression cost
    cost = -(1 / m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    total_cost = cost + reg_term
    return total_cost

# Gradient computation with regularization
def compute_gradient_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for logistic regression with regularization.
    """
    m, n = X.shape
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)

    dj_db = (1 / m) * np.sum(f_wb - y)
    dj_dw = (1 / m) * np.dot(X.T, f_wb - y) + (lambda_ / m) * w  # Regularization for w

    return dj_db, dj_dw

# Gradient descent with regularization
def gradient_descent_reg(X, y, w, b, alpha, num_iters, lambda_):
    """
    Performs gradient descent to learn w and b.
    """
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_reg(X, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        # Print cost every 100 iterations
        if i % 100 == 0 or i == num_iters - 1:
            cost = compute_cost_reg(X, y, w, b, lambda_)
            print(f"Iteration {i}: Cost {cost:.4f}")
    
    return w, b

# Prediction function
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters.
    """
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    return (f_wb >= 0.5).astype(int)

# Generate simple synthetic data
np.random.seed(1)
X = np.random.randn(100, 2)  # 100 examples, 2 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linearly separable data
y = y.reshape(-1, 1)

# Visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral)
plt.title("Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Initialize parameters
m, n = X.shape
w = np.zeros((n,))  # Initialize weights
b = 0.              # Initialize bias
alpha = 0.1         # Learning rate
num_iters = 1000    # Number of iterations
lambda_ = 0.1       # Regularization parameter

# Train the model
w, b = gradient_descent_reg(X, y, w, b, alpha, num_iters, lambda_)

# Plot decision boundary
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                       np.arange(x2_min, x2_max, 0.1))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = sigmoid(np.dot(grid, w) + b).reshape(xx1.shape)

plt.contourf(xx1, xx2, probs, levels=[0, 0.5, 1], alpha=0.8, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolor="k", cmap=plt.cm.Spectral)
plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
