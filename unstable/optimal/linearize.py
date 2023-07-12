# lqg.py
# Johannes Kaisinger, July 2023
#
# Demonstrate a linearization
#
# [1] 

import control as ct
import numpy as np

# Mass spring damper system
m = 1.; # mass
k = 0.5; # spring
c = 0.1; # damper

# Nonlinear mass spring damper
def mass_spring_damper_dynamics(t, x, u, params):
    """Internal state of adaptive controller, f(t,x,u;p)"""
    x1 = x[0]
    x2 = x[1]

    u1 = u[0]

    # System dynamics
    d_x1 = x2
    d_x2 = - k/m*x1**2 + - c/m*x2 + 1/m*u1
    return [d_x1, d_x2]


io_msd = ct.nlsys(
    mass_spring_damper_dynamics,
    None,
    inputs=1,
    outputs=2,
    states=2,
    name='nl_msd',
    dt=0
)

# Linearized system for controller synthesis
sys =  ct.linearize(io_msd, [0, 0], [0])
print(type(sys))