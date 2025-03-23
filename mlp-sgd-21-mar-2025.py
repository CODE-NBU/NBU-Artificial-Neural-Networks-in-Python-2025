import numpy as np
from sklearn.neural_network import MLPRegressor

# XOR input and output.
X = np.array([
	[-0.95, -0.95],
	[-0.95, +0.95],
	[+0.95, -0.95],
	[+0.95, +0.95]
])

# XOR output (continuous target values).
y = np.array([-0.95, +0.95, +0.95, -0.95])

# Create an MLPRegressor with 'tanh' activation function.
mlp = MLPRegressor(
	hidden_layer_sizes=(2,),
	activation='tanh',
	solver='sgd',
	tol=1e-9,
	n_iter_no_change=10000,
	early_stopping=False,
	max_iter=10000000000,
	verbose=True
)

# Train the model.
mlp.fit(X, y)

# Make predictions.
predictions = mlp.predict(X)

# Output predictions.
print("Input:\n", X)
print("Predictions:\n", predictions)
print("Actual:\n", y)

# MLP topology information.
print("\n=== MLP Topology ===")

# Number of layers (input + hidden + output).
print(f"Number of layers (including input and output): {mlp.n_layers_}")

# Number of outputs.
print(f"Number of outputs: {mlp.n_outputs_}")

# Hidden layer sizes (specified).
print(f"Hidden layer sizes (specified): {mlp.hidden_layer_sizes}")

# Weight matrices.
print("\nWeight matrices (mlp.coefs_):")
for idx, coef in enumerate(mlp.coefs_):
	print(f" Layer {idx} weights shape: {coef.shape}")
	print(coef)

# Bias vectors.
print("\nBias vectors (mlp.intercepts_):")
for idx, intercept in enumerate(mlp.intercepts_):
	print(f" Layer {idx} biases shape: {intercept.shape}")
	print(intercept)

