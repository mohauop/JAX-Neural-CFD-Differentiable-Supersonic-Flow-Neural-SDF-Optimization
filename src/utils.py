import jax.numpy as jnp

def cons_to_prim_1d(U, gamma=1.4):
    """Converts 1D conservative state [rho, rho*u, E] to primitive vars."""
    rho = jnp.maximum(U[..., 0], 1e-6)
    u = U[..., 1] / rho
    E = U[..., 2]
    p = jnp.maximum((gamma - 1.0) * (E - 0.5 * rho * u**2), 1e-6)
    c = jnp.sqrt(gamma * p / rho)
    return rho, u, p, c

def cons_to_prim_2d(U, gamma=1.4):
    """Converts 2D conservative state [rho, rho*u, rho*v, E] to primitive vars."""
    rho = jnp.maximum(U[..., 0], 1e-6)
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    E = U[..., 3]
    p = jnp.maximum((gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2)), 1e-6)
    c = jnp.sqrt(gamma * p / rho)
    return rho, u, v, p, c

def flux_x(U, gamma=1.4):
    """Computes physical flux in X-direction for 2D Euler equations."""
    rho, u, v, p, _ = cons_to_prim_2d(U, gamma)
    return jnp.stack([U[..., 1], U[..., 1] * u + p, U[..., 1] * v, u * (U[..., 3] + p)], axis=-1)

def flux_y(U, gamma=1.4):
    """Computes physical flux in Y-direction for 2D Euler equations."""
    rho, u, v, p, _ = cons_to_prim_2d(U, gamma)
    return jnp.stack([U[..., 2], U[..., 1] * v, U[..., 2] * v + p, v * (U[..., 3] + p)], axis=-1)