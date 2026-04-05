# Physics-Informed Neural Network (PINN) for the Fisher–KPP Equation

This project implements a Physics-Informed Neural Network (PINN) to solve the Fisher–KPP equation:

\[
u_t = D u_{xx} + r u (1 - u)
\]

It uses a PINN to solve the forward problem, and also attempts the inverse problem to recover parameters D (diffusion) and r (growth rate) by incorporating them as parameters in the network. A finite difference solver is used to serve as a reference solution for evaluating PINN performance.

## Dependencies

```bash
pip install torch numpy matplotlib
