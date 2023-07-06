# mrac_siso_indirect_lya_rule_statespace.py
# Johannes Kaisinger, June 2023
#
# Demonstrate a indirect MRAC example for a SISO plant using Lyapunov rule.
# Notation as in [2].
#
# [1] K. J. Aström & B. Wittenmark "Adaptive Control" Second Edition, 2008.
#
# [2] Nhan T. Nguyen "Model-Reference Adaptive Control", 2018.


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import control as ct

# linear system
A = 1.
B = 3.
C = 1.
D = 0.

G_plant_ss = ct.StateSpace(A,B,C,D)

# io_plant_model
io_plant = ct.LinearIOSystem(
    G_plant_ss,
    inputs=('u'),
    outputs=('x'),
    states=('x'),
    name='plant'
)

# linear system
Am = -4.
Bm = 4.
Cm = 1.
Dm = 0.

G_model_ss = ct.StateSpace(Am,Bm,Cm,Dm)

# io_ref_model
io_ref_model = ct.LinearIOSystem(
    G_model_ss,
    inputs=('r'),
    outputs=('xm'),
    states=('xm'),
    name='ref_model'
)

kx_star = (Am - A)/B
kr_star = Bm/B
print(f'{kx_star = }')
print(f'{kr_star = }')

def adaptive_controller_state(_t, xc, uc, params):
    """Internal state of adaptive controller, f(t,x,u;p)"""
    
    # Parameters
    gam_a = params["gam_a"]
    gam_b = params["gam_b"]
    Am = params["Am"]
    Bm = params["Bm"]
    signB = params["signB"]
    b0 = params["b0"]
    
    # Controller inputs
    r = uc[0]
    x = uc[1]
    xm = uc[2]
    
    e1 = xm - x

    # Algebraic relationships
    Ad = x1 = xc[0] # Ad
    Bd = x2 = xc[1] # Bd
    
    # Controller dynamics
    d_x1 = - gam_a*e1*x
    if (np.abs(Bd) > b0):
        kx = (Am-Ad)/Bd
        kr = Bm/Bd
        # Control law
        u = kx*x + kr*r
        d_x2 = - gam_b*e1*u
    elif (np.abs(Bd) == b0):
        kx = (Am-Ad)/Bd
        kr = Bm/Bd
        # Control law
        up = kx*x + kr*r
        if (gam_b*e1*up*signB >= 0):
            d_x2 = - gam_b*e1*u
        else:
            d_x2 = 0
    else:
        d_x2 = 0
    
    return [d_x1, d_x2]

def adaptive_controller_output(_t, xc, uc, params):
    """Algebraic output from adaptive controller, g(t,x,u;p)"""
    
    # Parameters
    Am = params["Am"]
    Bm = params["Bm"]

    # Controller inputs
    r = uc[0]
    x = uc[1]
    xm = uc[2]
    
    # Plant parameter estimates
    Ad = xc[0]
    Bd = xc[1]

    # Controller state
    kx = (Am-Ad)/Bd
    kr = Bm/Bd

    # Control law
    u = kx*x + kr*r

    return [u, kr, kx, Ad, Bd]

b0=1e-3
params={"gam_a":1., "gam_b":1., "Am":Am, "Bm":Bm, "signB":np.sign(B), "b0":b0}

io_controller = ct.NonlinearIOSystem(
    adaptive_controller_state,
    adaptive_controller_output,
    inputs=('r', 'x', 'xm'),
    outputs=('u', 'kr', 'kx', 'Ad', 'Bd'),
    states=2,
    params=params,
    name='control',
    dt=0
)

io_closed = ct.InterconnectedSystem(
    [io_plant, io_ref_model, io_controller],
    connections=[
        ['plant.u', 'control.u'],
        ['control.x', 'plant.x'],
        ['control.xm', 'ref_model.xm']
    ],
    inplist=['control.r', 'ref_model.r'],
    outlist=['plant.x', 'ref_model.xm', 'control.u', 'control.kr', 'control.kx', 'control.Ad', 'control.Bd'],
    dt=0
)

# Set simulation duration and time steps
Tend = 100
dt = 0.1

# Define simulation time 
t_vec = np.arange(0, Tend, dt)

# Define control reference input
r_vec = np.zeros((2, len(t_vec)))

rect = signal.square(2 * np.pi * 0.05 * t_vec)
sin = np.sin(2 * np.pi * 0.05 * t_vec) + np.sin(2 * np.pi * 0.5 * t_vec)
r_vec[0, :] = rect
r_vec[1, :] = r_vec[0, :]

plt.figure(figsize=(16,8))
plt.plot(t_vec, r_vec[0,:])
plt.title(r'reference input $r$')
plt.show()

# Set initial conditions, io_closed
X0 = np.zeros((4, 1))
X0[0] = 0 # state of plant, (x)
X0[1] = 0 # state of ref_model, (xm)
X0[2] = 0 # state of controller, (kr)
X0[3] = b0*1000 # state of controller, (kx)

# Simulate the system with different gammas
tout1, yout1 = ct.input_output_response(io_closed, t_vec, r_vec, X0, params={"gam_a":0.2, "gam_b":0.2})
tout2, yout2 = ct.input_output_response(io_closed, t_vec, r_vec, X0, params={"gam_a":1.0, "gam_b":1.0})
tout3, yout3 = ct.input_output_response(io_closed, t_vec, r_vec, X0, params={"gam_a":5.0, "gam_b":5.0})

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.plot(tout1, yout1[0,:], label=r'$y_{\gamma = 0.2}$')
plt.plot(tout2, yout2[0,:], label=r'$y_{\gamma = 1.0}$')
plt.plot(tout2, yout2[0,:], label=r'$y_{\gamma = 5.0}$')
plt.plot(tout1, yout1[1,:] ,label=r'$y_{m}$', linestyle='--')
plt.legend()
plt.title('system response $x, (x_m)$')
plt.subplot(2,1,2)
plt.plot(tout1, yout1[2,:], label=r'$u$')
plt.plot(tout2, yout2[2,:], label=r'$u$')
plt.plot(tout3, yout3[2,:], label=r'$u$')
plt.legend(loc=4)
plt.title(r'control $u$')
plt.show()

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.plot(tout1, yout1[5,:], label=r'$\gamma = 0.2$')
plt.plot(tout2, yout2[5,:], label=r'$\gamma = 1.0$')
plt.plot(tout3, yout3[5,:], label=r'$\gamma = 5.0$')
plt.hlines(A, 0, Tend, label=r'$A$', color='black', linestyle='--')
plt.legend(loc=4)
plt.title(r'system parameter $\hat{A}$')
plt.subplot(2,1,2)
plt.plot(tout1, yout1[6,:], label=r'$\gamma = 0.2$')
plt.plot(tout2, yout2[6,:], label=r'$\gamma = 1.0$')
plt.plot(tout3, yout3[6,:], label=r'$\gamma = 5.0$')
plt.hlines(B, 0, Tend, label=r'$B$', color='black', linestyle='--')
plt.legend(loc=4)
plt.title(r'system parameter $\hat{B}$')
plt.show()

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.plot(tout1, yout1[4,:], label=r'$\gamma = 0.2$')
plt.plot(tout2, yout2[4,:], label=r'$\gamma = 1.0$')
plt.plot(tout3, yout3[4,:], label=r'$\gamma = 5.0$')
plt.hlines(kx_star, 0, Tend, label=r'$k_x^{\ast}$', color='black', linestyle='--')
plt.legend(loc=4)
plt.title(r'control gain $k_x$ (feedback)')
plt.subplot(2,1,2)
plt.plot(tout1, yout1[3,:], label=r'$\gamma = 0.2$')
plt.plot(tout2, yout2[3,:], label=r'$\gamma = 1.0$')
plt.plot(tout3, yout3[3,:], label=r'$\gamma = 5.0$')
plt.hlines(kr_star, 0, Tend, label=r'$k_r^{\ast}$', color='black', linestyle='--')
plt.legend(loc=4)
plt.title(r'control gain $k_r$ (feedforward)')
plt.show()
