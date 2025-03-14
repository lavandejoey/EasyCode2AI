import numpy as np
import matplotlib.pyplot as plt

EPOCH = 10
PLOT_INTERVAL = 2
DATA_SIZE = 50

# Generate synthetic 2D regression dataset
np.random.seed(0)
X = 2 * np.random.rand(DATA_SIZE, 1)
y = 4 + 3 * X + np.random.randn(DATA_SIZE, 1)  # y = 4 + 3x + noise

# Add bias term (x0 = 1)
X_b = np.c_[np.ones((DATA_SIZE, 1)), X]  # shape (DATA_SIZE, 2)


# Perceptron model for regression
class PerceptronRegressor:
    def __init__(self, lr=0.1):
        self.lr = lr
        # two weights (bias + x weight)
        self.theta = np.random.randn(2, 1)

    def predict(self, X):
        return X @ self.theta

    def fit(self, X, y, epochs=EPOCH, plot_interval=PLOT_INTERVAL):
        plt.figure(figsize=(10, 6))
        for epoch in range(1, epochs + 1):
            y_pred = self.predict(X)
            error = y_pred - y
            gradients = 2 / len(X) * X.T @ error
            self.theta -= self.lr * gradients

            # Plot every `plot_interval` epochs
            if epoch % plot_interval == 0 or epoch == 1:
                x_line = np.array([[1, 0], [1, 2]])
                y_line = self.predict(x_line)
                plt.plot(x_line[:, 1], y_line, label=f'Epoch {epoch}')

        # Final plot settings
        plt.scatter(X[:, 1], y, color='black', label='Data', alpha=0.6)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Learning Progress of Perceptron (Regression)')
        plt.legend()
        plt.grid(True)
        plt.show()


# Create and train the perceptron regressor
perceptron = PerceptronRegressor(lr=0.05)
perceptron.fit(X_b, y)
