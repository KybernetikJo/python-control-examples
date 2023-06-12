import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import control as ct
print(ct.__version__)
import slycot
print(slycot.__version__)

# linear system, 
Ap = -1.
Bp = 0.5
Cp = 1
Dp = 0

G_plant_ss = ct.StateSpace(Ap,Bp,Cp,Dp)

# io_plant_model
io_plant = ct.LinearIOSystem(
    G_plant_ss,
    inputs=('up'),
    outputs=('xp'),
    states=('xp'),
    name='plant'
)

# linear system
Ar = -2
Br = 2
Cr = 1
Dr = 0

G_model_ss = ct.StateSpace(Ar,Br,Cr,Dr)

# io_ref_model
io_ref_model = ct.LinearIOSystem(
    G_model_ss,
    inputs=('ur'),
    outputs=('xr'),
    states=('xr'),
    name='ref_model'
)

kr_star = (Br)/Bp
print(f"Optimal value for {kr_star = }")
kx_star = (Ar-Ap)/Bp
print(f"Optimal value for {kx_star = }")

def adaptive_controller_state(t, x, u, params):
    """Internal state of adpative controller, f(t,x,u;p)"""
    
    # parameters
    gam = params["gam"]
    Ar = params["Ar"]
    Br = params["Br"]
    signb = params["signb"]

    # controller inputs
    ur = u[0]
    xp = u[1]
    xr = u[2]

    # controller states
    x1 = x[0] #
    x2 = x[1] # kr
    x3 = x[2] # 
    x4 = x[3] # kx
    
    # algebraic relationships
    e = (xr - xp)

    d_x1 = Ar*x1 + Ar*ur
    d_x2 = - gam*(x1)*e*signb
    d_x3 = Ar*x3 + Ar*xp
    d_x4 = - gam*(x3)*e*signb

    return [d_x1, d_x2, d_x3, d_x4]

def adaptive_controller_output(_t, x, u, params):
    """Algebraic output from adaptive controller, g(t,x,u;p)"""

    # controller inputs
    ur = u[0]
    xp = u[1]
    xr = u[2]
    
    # controller state
    kr = x[1]
    kx = x[3]
    
    # control law
    up = + kx*xp + kr*ur

    return [up, kx, kr]

params={"gam":1, "Ar":Ar, "Br":Br, "signb":np.sign(Bp)}

io_controller = ct.NonlinearIOSystem(
    adaptive_controller_state,
    adaptive_controller_output,
    inputs=3,
    outputs=('up', 'kx', 'kr'),
    states=4,
    params=params,
    name='control',
    dt=0
)

io_closed = ct.InterconnectedSystem(
    (io_plant, io_ref_model, io_controller),
    connections=[
        ['plant.up', 'control.up'],
        ['control.u[1]', 'plant.xp'],
        ['control.u[2]', 'ref_model.xr']
    ],
    inplist=['control.u[0]', 'ref_model.ur'],
    outlist=['plant.xp', 'ref_model.xr', 'control.up', 'control.kx', 'control.kr'],
    dt=0
)

# set initial conditions
X0 = np.zeros((6, 1))
X0[0] = 0 # state of plant, (xp)
X0[1] = 0 # state of ref_model, (xr)
X0[2] = 0 # state of controller, (kr_dot)
X0[3] = 0 # state of controller, (kr)
X0[4] = 0 # state of controller, (kx_dot)
X0[5] = 0 # state of controller, (kx)

# set simulation duration and time steps
Tend = 100
dt = 0.1

# define simulation time span 
t_vec = np.arange(0, Tend, dt)

# define control reference input
ur_vec = np.zeros((2, len(t_vec)))
square = signal.square(2 * np.pi * 0.05 * t_vec)
ur_vec[0, :] = square
ur_vec[1, :] = ur_vec[0, :]

plt.figure(figsize=(16,8))
plt.plot(t_vec, ur_vec[0,:])
plt.title(r'reference input $u_r$')
plt.show()

# simulate the system, with different gammas
tout1, yout1 = ct.input_output_response(io_closed, t_vec, ur_vec, X0, params={"gam":0.2})
tout2, yout2 = ct.input_output_response(io_closed, t_vec, ur_vec, X0, params={"gam":1.0})
tout3, yout3 = ct.input_output_response(io_closed, t_vec, ur_vec, X0, params={"gam":5.0})

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.plot(tout1, yout1[0,:], label=r'$y_{p, \gamma = 0.2}$')
plt.plot(tout2, yout2[0,:], label=r'$y_{p, \gamma = 1.0}$')
plt.plot(tout2, yout3[0,:], label=r'$y_{p, \gamma = 5.0}$')
plt.plot(tout1, yout1[1,:] ,label=r'$y_{r}$')
plt.legend(fontsize=14)
plt.title(r'system response $y_p, (y_r)$')
plt.subplot(2,1,2)
plt.plot(tout1, yout1[2,:], label=r'$u_{p, \gamma = 0.2}$')
plt.plot(tout2, yout2[2,:], label=r'$u_{p, \gamma = 1.0}$')
plt.plot(tout3, yout3[2,:], label=r'$u_{p, \gamma = 5.0}$')
plt.legend(loc=4, fontsize=14)
plt.title(r'control $u_p$')
plt.show()

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.plot(tout1, yout1[3,:], label=r'$k_{x, \gamma = 0.2}$')
plt.plot(tout2, yout2[3,:], label=r'$k_{x, \gamma = 1.0}$')
plt.plot(tout3, yout3[3,:], label=r'$k_{x, \gamma = 5.0}$')
plt.hlines(kx_star, 0, Tend, label=r'$k_x^{\ast}$', color='black', linestyle='--')
plt.legend(loc=4, fontsize=14)
plt.title(r'control gain $k_x$')
plt.subplot(2,1,2)
plt.plot(tout1, yout1[4,:], label=r'$k_{r, \gamma = 0.2}$')
plt.plot(tout2, yout2[4,:], label=r'$k_{r, \gamma = 1.0}$')
plt.plot(tout3, yout3[4,:], label=r'$k_{r, \gamma = 5.0}$')
plt.hlines(kr_star, 0, Tend, label=r'$k_r^{\ast}$', color='black', linestyle='--')
plt.legend(loc=4, fontsize=14)
plt.title(r'control gain $k_r$')
plt.show()