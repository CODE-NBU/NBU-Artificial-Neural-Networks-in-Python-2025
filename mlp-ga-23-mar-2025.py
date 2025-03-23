import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

# XOR input and output.
X = np.array([
    [-0.95, -0.95],
    [-0.95, +0.95],
    [+0.95, -0.95],
    [+0.95, +0.95]
])

y = np.array([-0.95, +0.95, +0.95, -0.95])

# Create MLPRegressor (weights will be overridden).
mlp = MLPRegressor(
    hidden_layer_sizes=(2,),
    activation='tanh',
    # Will be bypassed.
    solver='sgd',
    tol=1e-9,
    n_iter_no_change=10000,
    early_stopping=False,
    # Train once just to initialize structure.
    max_iter=1,
    verbose=False
)

mlp.fit(X, y)

# Helpers to flatten/unflatten weights.
def flatten_weights(coefs, intercepts):
    flat = np.concatenate([w.flatten() for w in coefs] + [b.flatten() for b in intercepts])
    return flat

def unflatten_weights(flat_array, mlp_structure):
    coefs = []
    intercepts = []
    idx = 0
    for shape in mlp_structure['coefs_shapes']:
        size = np.prod(shape)
        coefs.append(flat_array[idx:idx+size].reshape(shape))
        idx += size
    for shape in mlp_structure['intercepts_shapes']:
        size = np.prod(shape)
        intercepts.append(flat_array[idx:idx+size].reshape(shape))
        idx += size
    return coefs, intercepts

# MLP structure info.
mlp_structure = {
    'coefs_shapes': [w.shape for w in mlp.coefs_],
    'intercepts_shapes': [b.shape for b in mlp.intercepts_]
}

initial_weights = flatten_weights(mlp.coefs_, mlp.intercepts_)
num_weights = initial_weights.size

# Fitness function: minimize mean squared error.
def eval_mlp(individual):
    coefs, intercepts = unflatten_weights(np.array(individual), mlp_structure)
    mlp.coefs_ = coefs
    mlp.intercepts_ = intercepts
    
    predictions = mlp.predict(X)
    mse = mean_squared_error(y, predictions)
    
    # Return as a tuple (DEAP expects fitness as a tuple).
    return (mse,)

# Genetic Algorithm setup.

# Minimize MSE.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_mlp)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the Genetic Algorithm with real-time plot.
population = toolbox.population(n=50)

NGEN = 1000

# Prepare live plotting.
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label="Best MSE")
ax.set_xlabel('Generation')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Genetic Algorithm - MLPRegressor XOR')
ax.legend()
y_vals = []

print("Evolving MLP weights with Genetic Algorithm...\n")
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    
    fits = toolbox.map(toolbox.evaluate, offspring)
    
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    
    population = toolbox.select(offspring, k=len(population))
    
    # Best individual in current generation.
    top_ind = tools.selBest(population, k=1)[0]
    best_mse = top_ind.fitness.values[0]
    
    y_vals.append(best_mse)
    
    # Update live plot.
    line.set_xdata(np.arange(len(y_vals)))
    line.set_ydata(y_vals)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    print(f"Gen {gen}: Best MSE = {best_mse:.6f}")

# Apply best weights to MLP.
best_weights = np.array(top_ind)
best_coefs, best_intercepts = unflatten_weights(best_weights, mlp_structure)
mlp.coefs_ = best_coefs
mlp.intercepts_ = best_intercepts

# Final prediction and evaluation.
predictions = mlp.predict(X)
final_mse = mean_squared_error(y, predictions)

# Stop live plot updates and show static plot.
plt.ioff()
plt.show()

print("\n=== Final Results ===")
print("Input:\n", X)
print("Predictions:\n", predictions)
print("Actual:\n", y)
print(f"\nFinal Mean Squared Error: {final_mse:.6f}")

# MLP topology and weights.
print("\n=== MLP Topology ===")
print(f"Number of layers (including input and output): {mlp.n_layers_}")
print(f"Number of outputs: {mlp.n_outputs_}")
print(f"Hidden layer sizes (specified): {mlp.hidden_layer_sizes}")

print("\nWeight matrices (mlp.coefs_):")
for idx, coef in enumerate(mlp.coefs_):
    print(f" Layer {idx} weights shape: {coef.shape}")
    print(coef)

print("\nBias vectors (mlp.intercepts_):")
for idx, intercept in enumerate(mlp.intercepts_):
    print(f" Layer {idx} biases shape: {intercept.shape}")
    print(intercept)

