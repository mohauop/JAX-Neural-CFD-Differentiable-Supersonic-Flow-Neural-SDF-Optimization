import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from src.solvers import step_2d
from src.neural_sdf import init_sdf_network, create_neural_mask
from src.utils import cons_to_prim_2d

def run_realtime_wind_tunnel():
    N = 180
    x, y = jnp.linspace(0, 1, N), jnp.linspace(0, 1, N)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    nn_params = init_sdf_network(jax.random.PRNGKey(123))

    @jax.jit
    def advance_frame(U_curr, angle=0.0, steps_per_frame=15):
        cx, cy = 0.3, 0.5
        X_c, Y_c = X - cx, Y - cy
        X_rot = X_c * jnp.cos(angle) - Y_c * jnp.sin(angle) + cx
        Y_rot = X_c * jnp.sin(angle) + Y_c * jnp.cos(angle) + cy
        
        mask = create_neural_mask(nn_params, X_rot, Y_rot)
        
        def scan_fn(U, _):
            return step_2d(U, mask, dx=0.005, dy=0.005, dt=0.0008), None
        
        U_next, _ = jax.lax.scan(scan_fn, U_curr, None, length=steps_per_frame)
        return U_next, mask

    # Initial Mach 1.5 State
    U = jnp.zeros((N, N, 4))
    U = U.at[..., 0].set(1.0)
    U = U.at[..., 1].set(1.5)
    U = U.at[..., 3].set(1.0 / (1.4 - 1.0) + 0.5 * 1.0 * 1.5**2)

    angle = 0.0
    print("Running Real-Time Interactive Wind Tunnel!")
    print("Controls: 'A' = Rotate Left | 'D' = Rotate Right | 'Q' = Quit")

    while True:
        U, current_mask = advance_frame(U, angle=angle)
        _, _, _, p, _ = cons_to_prim_2d(U)
        
        p_norm = np.array((p - p.min()) / (p.max() - p.min() + 1e-6) * 255, dtype=np.uint8)
        frame = cv2.applyColorMap(p_norm, cv2.COLORMAP_JET)
        
        mask_np = np.array(current_mask) < 0.5
        frame[mask_np] = [0, 0, 0]

        cv2.imshow("JAX Real-Time Supersonic CFD", cv2.resize(frame, (600, 600)))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            angle -= 0.08
        elif key == ord('d'):
            angle += 0.08
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_wind_tunnel()