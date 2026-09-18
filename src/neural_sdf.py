import jax
import jax.numpy as jnp

def init_sdf_network(key):
    """Initializes weights and biases for 2->32->32->1 Coordinate MLP."""
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        'w1': jax.random.normal(k1, (2, 32)) * jnp.sqrt(2.0 / 2),
        'b1': jnp.zeros((32,)),
        'w2': jax.random.normal(k2, (32, 32)) * jnp.sqrt(2.0 / 32),
        'b2': jnp.zeros((32,)),
        'w3': jax.random.normal(k3, (32, 1)) * jnp.sqrt(2.0 / 32),
        'b3': jnp.zeros((1,))
    }
    return params

def sdf_forward(params, coords):
    """Evaluates MLP forward pass to predict SDF value at input coordinates."""
    h = jnp.dot(coords, params['w1']) + params['b1']
    h = jax.nn.silu(h)
    h = jnp.dot(h, params['w2']) + params['b2']
    h = jax.nn.silu(h)
    sdf = jnp.dot(h, params['w3']) + params['b3']
    return sdf.squeeze(-1)

def create_neural_mask(params, X, Y, cx=0.3, cy=0.5, sharpness=30.0):
    """Generates continuous fluid mask M(x,y) in [0, 1] using Sigmoid soft level-set."""
    grid_coords = jnp.stack([X - cx, Y - cy], axis=-1)
    sdf_vals = sdf_forward(params, grid_coords)
    return jax.nn.sigmoid(sdf_vals * sharpness)