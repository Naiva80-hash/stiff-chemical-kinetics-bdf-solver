# Adaptive BDF Solver for Stiff Chemical Reaction Kinetics

This project implements an implicit numerical solver for a stiff
seven-species chemical reaction system.

The project was developed as part of an Advanced Numerical Methods
in Chemical Engineering course.

## Overview

The reaction system consists of seven concentration states:

- CA
- CB
- CP
- CQ
- CX
- CY
- CZ

The large differences between kinetic rate constants make the resulting
ODE system stiff and therefore challenging for standard explicit
integration methods.

The objective is to accurately simulate the concentration dynamics over
both short and long time scales.

## Numerical Methods

### Variable-Step BDF2

A variable-step second-order Backward Differentiation Formula (BDF2)
is implemented for integration of the stiff ODE system.

A BDF1 step is used for initialization, followed by adaptive BDF2 steps.

### Newton Solver

Each implicit BDF step produces a nonlinear algebraic system.

The system is solved using a custom Newton iteration featuring:

- Analytical Jacobian evaluation
- Backtracking line search
- Jacobian/LU reuse
- Jacobian rebuilding when convergence stalls

### Adaptive Time Stepping

The time step is automatically increased or reduced according to a
weighted error estimate.

Failed Newton iterations and excessive local errors trigger step rejection
and time-step reduction.

### Linear Algebra

A custom LU decomposition with partial pivoting is used to solve the
Newton linear systems.

## Validation

BDF residuals are evaluated after integration to verify consistency of
the numerical solution.

The solver is tested over long simulation horizons of up to 10,000 s.

## Technologies

- Python
- NumPy
- Matplotlib

## Running

Install dependencies:

```bash
pip install -r requirements.txt
