# JAX-Neural-CFD: Differentiable Supersonic Flow & Neural SDF Optimization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A pure-JAX, GPU-accelerated 1D/2D Euler equations solver coupled with continuous Neural Signed Distance Fields (Neural SDFs) for end-to-end differentiable aerodynamic shape optimization and real-time interactive fluid simulation.

---

## Key Features

* **End-to-End Differentiable CFD:** Automatic differentiation through full time-stepping schemes via `jax.lax.scan` and reverse-mode AD (`jax.grad`).
* **Neural SDF Geometry:** Continuous geometry parameterization using Coordinate MLPs ($2 \rightarrow 32 \rightarrow 32 \rightarrow 1$) to eliminate manual mesh generation.
* **Pure JAX / Zero-Copy Pipelines:** Runs entire physics execution, boundary conditions, and gradient updates fully in hardware memory with zero CPU-GPU transfer bottlenecks.
* **Real-Time Interactive GUI:** Integrated OpenCV visualization engine achieving **60+ FPS** interactive flow simulations with real-time shape manipulation.
* **High-Throughput GPU Solver:** Achieves **>200 MLUPS** (Million Cell Updates Per Second) on modern GPUs using XLA compilation.

---

## Architecture Overview

```text
+-------------------+    +--------------------+    +----------------------+
| Coordinates: X,Y  |--> | Neural SDF Network |--> | Continuous Body Mask |
+-------------------+    +--------------------+    +----------------------+
                                                                  |
                                                                  v
+-------------------------------------------------------------------------+
|                  2D Euler JAX Solver (Rusanov Scheme)                   |
+-------------------------------------------------------------------------+
                                      ^
                                      | (Target Objective)
+-------------------------------------------------------------------------+
| Reverse AD (jax.grad) ===> Optimizes MLP Weights                        |
+-------------------------------------------------------------------------+
```

## Mathematical Formulation

### 1. Governing Physics: 2D Compressible Euler Equations
The fluid flow is governed by the non-linear, inviscid 2D Euler conservation laws for mass, momentum, and total energy:

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} + \frac{\partial \mathbf{G}(\mathbf{U})}{\partial y} = 0$$

$$\mathbf{U} = \begin{bmatrix} \rho \\ \rho u \\ \rho v \\ E \end{bmatrix}, \quad 
\mathbf{F}(\mathbf{U}) = \begin{bmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(E + p) \end{bmatrix}, \quad 
\mathbf{G}(\mathbf{U}) = \begin{bmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(E + p) \end{bmatrix}$$

where pressure $p$ is closed using the ideal gas equation of state ($\gamma = 1.4$):

$$p = (\gamma - 1) \left( E - \frac{1}{2} \rho (u^2 + v^2) \right)$$

### 2. Spatial Discretization: Rusanov Numerical Flux
Cell interface fluxes are evaluated using the local Lax-Friedrichs (Rusanov) scheme:

$$\mathbf{F}_{i+1/2} = \frac{1}{2} \left( \mathbf{F}(\mathbf{U}_L) + \mathbf{F}(\mathbf{U}_R) \right) - \frac{1}{2} s_{\max} (\mathbf{U}_R - \mathbf{U}_L)$$

where the maximum local wave speed is determined by $s_{\max} = \max(|u_L| + c_L, |u_R| + c_R)$ and speed of sound $c = \sqrt{\gamma p / \rho}$.

### 3. Geometry Representation: Neural Signed Distance Field (SDF)
Obstacle boundaries are defined as continuous level-sets parameterized by a Coordinate MLP $f_\theta(x, y)$:

$$M_\theta(x, y) = \sigma \Big( \alpha \cdot f_\theta(x - c_x, y - c_y) \Big)$$

where $\sigma(\cdot)$ is the Sigmoid activation function, $\alpha$ controls interface sharpness, and $M(x, y) \in [0, 1]$ smoothly enforces no-slip boundary constraints inside the time-stepping loop.

### 4. Differentiable Optimization Objective
The end-to-end loss minimizes total wave drag force along the flow direction while penalizing deviations from a target area $A_0$:

$$\mathcal{L}(\theta) = \int_{\Omega} p(x,y) \frac{\partial M_\theta}{\partial x} dx dy \;+\; \lambda \left( \iint_\Omega (1 - M_\theta(x,y)) dx dy - A_0 \right)^2$$

## Installation

Ensure you have a modern Python environment and JAX installed with appropriate GPU drivers (CUDA / ROCm).

```bash
# Clone repository
git clone [https://github.com/mohauop/JAX-Neural-CFD-Differentiable-Supersonic-Flow-Neural-SDF-Optimization.git](https://github.com/mohauop/JAX-Neural-CFD-Differentiable-Supersonic-Flow-Neural-SDF-Optimization.git)
cd jax-neural-cfd
```

# Install dependencies
```bash
pip install jax jaxlib optax opencv-python matplotlib numpy
```

# Quick Start
# 1. Differentiable Shape Optimization
Run end-to-end shape optimization to discover the minimal wave-drag profile in supersonic flow (Mach 1.5):

```bash
python examples/02_shape_optimization.py
```
# 2. Launch Interactive Real-Time GUI
Launch the real-time interactive OpenCV wind tunnel window to manipulate geometry live:

```Bash
python examples/interactive_wind_tunnel.py
Controls: Press A / D to rotate neural geometry live | Press Q to quit.
```
## Live Interactive Simulation

Try the interactive 2D solver directly in your browser (no installation required):
[**Launch Live CFD Simulation**](https://github.io/mohauop/JAX-Neural-CFD-Differentiable-Supersonic-Flow-Neural-SDF-Optimization/simulation.html)
## Benchmarks & Performance

*Tested on **NVIDIA RTX 3090** (CUDA 12.x, JAX 0.4.x) over 200 simulation time-steps:*

| Grid Resolution | Total Cells | Compile Time (XLA) | Execution Time | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **100 x 100** | 10,000 | ~180 ms | 2.1 ms | **95 MLUPS** |
| **200 x 200** | 40,000 | ~210 ms | 3.8 ms | **210 MLUPS** |
| **400 x 400** | 160,000 | ~350 ms | 14.2 ms | **225 MLUPS** |