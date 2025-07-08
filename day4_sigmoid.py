import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate data
z = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z)

# plot the sigmoid function
plt.plot(z, sigmoid_values, label='Sigmoid Function', color='blue')
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.ylabel("sigmoid(z)")
plt.grid()
plt.show()