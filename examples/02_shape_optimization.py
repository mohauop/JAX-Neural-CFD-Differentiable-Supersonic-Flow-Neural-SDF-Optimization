import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from src.solvers import simulate_2d_custom_mask
from src.neural_sdf import init_sdf_network, create_neural_mask
from src.utils import cons_to_prim_2d

def optimize_neural_shape():
    nx, ny = 100, 100
    x, y = jnp.linspace(0, 1, nx), jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    key = jax.random.PRNGKey(42)
    nn_params = init_sdf_network(key)
    optimizer = optax.adam(learning_rate=0.002)
    opt_state = optimizer.init(nn_params)

    def loss_fn(params):
        mask = create_neural_mask(params, X, Y)
        U_final = simulate_2d_custom_mask(mask, nx=nx, ny=ny, n_steps=40)
        _, _, _, p, _ = cons_to_prim_2d(U_final)
        
        # Calculate wave drag force along flow axis
        dmask_dx = jnp.gradient(mask, axis=0)
        drag_loss = jnp.sum(p * dmask_dx)
        
        # Penalty term to keep obstacle volume finite
        volume_penalty = 10.0 * (jnp.mean(1.0 - mask) - 0.05)**2
        return drag_loss + volume_penalty

    @jax.jit
    def train_step(params, state):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = optimizer.update(grads, state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss_val

    print("Starting Neural SDF Shape Optimization...")
    for iter_idx in range(15):
        nn_params, opt_state, loss = train_step(nn_params, opt_state)
        print(f"Iter {iter_idx+1:02d} | Loss: {loss:.5f}")

    # Plot final optimized shape
    opt_mask = create_neural_mask(nn_params, X, Y)
    plt.figure(figsize=(5, 5))
    plt.imshow(opt_mask.T, origin='lower', cmap='Blues_r', extent=[0, 1, 0, 1])
    plt.contour(opt_mask.T, levels=[0.5], colors='red', extent=[0, 1, 0, 1])
    plt.title("Optimized Neural SDF Shape (Min Wave Drag)")
    plt.savefig("optimized_shape.png")
    print("Optimization finished. Result saved as optimized_shape.png")

if __name__ == "__main__":
    optimize_neural_shape()