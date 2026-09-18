import jax
import jax.numpy as jnp
from src.utils import cons_to_prim_1d, cons_to_prim_2d, flux_x, flux_y

GAMMA = 1.4

def rusanov_1d(U_L, U_R):
    """Rusanov numerical flux for 1D Euler solver."""
    rho_L, u_L, p_L, c_L = cons_to_prim_1d(U_L, GAMMA)
    rho_R, u_R, p_R, c_R = cons_to_prim_1d(U_R, GAMMA)
    
    F_L = jnp.stack([U_L[..., 1], U_L[..., 1] * u_L + p_L, u_L * (U_L[..., 2] + p_L)], axis=-1)
    F_R = jnp.stack([U_R[..., 1], U_R[..., 1] * u_R + p_R, u_R * (U_R[..., 2] + p_R)], axis=-1)
    
    s_max = jnp.maximum(jnp.abs(u_L) + c_L, jnp.abs(u_R) + c_R)[..., None]
    return 0.5 * (F_L + F_R) - 0.5 * s_max * (U_R - U_L)

def step_1d(U, dx=0.01, dt=0.001):
    """Performs single time-step update for 1D Euler equations."""
    U_padded = jnp.pad(U, ((1, 1), (0, 0)), mode='edge')
    F = rusanov_1d(U_padded[:-1], U_padded[1:])
    return U + (dt / dx) * (F[:-1] - F[1:])

def rusanov_2d(U_L, U_R, axis):
    """Rusanov interface flux along spatial axis (0 for X, 1 for Y)."""
    if axis == 0:
        _, u_L, _, _, c_L = cons_to_prim_2d(U_L, GAMMA)
        _, u_R, _, _, c_R = cons_to_prim_2d(U_R, GAMMA)
        F_L, F_R = flux_x(U_L, GAMMA), flux_x(U_R, GAMMA)
        s_max = jnp.maximum(jnp.abs(u_L) + c_L, jnp.abs(u_R) + c_R)[..., None]
    else:
        _, _, v_L, _, c_L = cons_to_prim_2d(U_L, GAMMA)
        _, _, v_R, _, c_R = cons_to_prim_2d(U_R, GAMMA)
        F_L, F_R = flux_y(U_L, GAMMA), flux_y(U_R, GAMMA)
        s_max = jnp.maximum(jnp.abs(v_L) + c_L, jnp.abs(v_R) + c_R)[..., None]

    return 0.5 * (F_L + F_R) - 0.5 * s_max * (U_R - U_L)

def step_2d(U, mask, dx=0.005, dy=0.005, dt=0.0008):
    """Performs single 2D finite-volume update step with spatial mask penalty."""
    U_px = jnp.pad(U, ((1, 1), (0, 0), (0, 0)), mode='edge')
    U_py = jnp.pad(U, ((0, 0), (1, 1), (0, 0)), mode='edge')

    Fx = rusanov_2d(U_px[:-1, :], U_px[1:, :], axis=0)
    Fy = rusanov_2d(U_py[:, :-1], U_py[:, 1:], axis=1)

    U_next = U + (dt / dx) * (Fx[:-1, :] - Fx[1:, :]) + (dt / dy) * (Fy[:, :-1] - Fy[:, 1:])

    rho, u, v, p, _ = cons_to_prim_2d(U_next, GAMMA)
    u_masked = u * mask
    v_masked = v * mask
    E_masked = p / (GAMMA - 1.0) + 0.5 * rho * (u_masked**2 + v_masked**2)

    return jnp.stack([rho, rho * u_masked, rho * v_masked, E_masked], axis=-1)

def simulate_2d_custom_mask(mask, nx=200, ny=200, n_steps=60, dx=0.005, dy=0.005, dt=0.0008, mach=1.5):
    """Executes full 2D CFD simulation around given spatial obstacle mask."""
    rho_in, u_in, v_in, p_in = 1.0, mach, 0.0, 1.0
    E_in = p_in / (GAMMA - 1.0) + 0.5 * rho_in * (u_in**2 + v_in**2)
    
    U_init = jnp.zeros((nx, ny, 4))
    U_init = U_init.at[..., 0].set(rho_in)
    U_init = U_init.at[..., 1].set(rho_in * u_in)
    U_init = U_init.at[..., 3].set(E_in)

    def scan_body(U_curr, _):
        U_next = step_2d(U_curr, mask, dx=dx, dy=dy, dt=dt)
        return U_next, None

    U_final, _ = jax.lax.scan(scan_body, U_init, None, length=n_steps)
    return U_final