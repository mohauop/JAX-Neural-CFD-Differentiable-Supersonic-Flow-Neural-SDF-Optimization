import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.solvers import step_1d
from src.utils import cons_to_prim_1d

def run_sod_benchmark():
    nx = 200
    n_steps = 150
    dx, dt = 1.0 / nx, 0.001

    # Sod shock tube initial conditions (Left state high pressure, Right state low pressure)
    U_init = jnp.zeros((nx, 3))
    
    # Left state (x <= 0.5)
    left_mask = jnp.linspace(0, 1, nx) <= 0.5
    rho_init = jnp.where(left_mask, 1.0, 0.125)
    p_init = jnp.where(left_mask, 1.0, 0.1)
    
    U_init = U_init.at[:, 0].set(rho_init)
    U_init = U_init.at[:, 2].set(p_init / (1.4 - 1.0))

    def scan_step(U, _):
        return step_1d(U, dx=dx, dt=dt), None

    U_final, _ = jax.lax.scan(scan_step, U_init, None, length=n_steps)
    rho, u, p, _ = cons_to_prim_1d(U_final)

    x = jnp.linspace(0, 1, nx)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    axes[0].plot(x, rho, 'b-', label='Density (rho)')
    axes[0].set_title("Density")
    axes[1].plot(x, u, 'g-', label='Velocity (u)')
    axes[1].set_title("Velocity")
    axes[2].plot(x, p, 'r-', label='Pressure (p)')
    axes[2].set_title("Pressure")

    for ax in axes:
        ax.grid(True)
    plt.suptitle("1D Sod Shock Tube Benchmark (Pure JAX)", fontsize=12)
    plt.tight_layout()
    plt.savefig("sod_shock_tube_result.png")
    print("Sod Shock Tube simulation completed successfully! Plot saved as sod_shock_tube_result.png")

if __name__ == "__main__":
    run_sod_benchmark()