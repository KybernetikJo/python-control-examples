import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import control as ct

# linear system
Ap = -1.
Bp = 0.5
Cp = 1.
Dp = 0.

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
Am = -2.
Bm = 2.
Cm = 1.
Dm = 0.

G_model_ss = ct.StateSpace(Am,Bm,Cm,Dm)

# io_ref_model
io_ref_model = ct.LinearIOSystem(
    G_model_ss,
    inputs=('um'),
    outputs=('xm'),
    states=('xm'),
    name='ref_model'
)

k_star = (Am - Ap)/Bp
l_star = Bm/Bp
print(f'{k_star = }')
print(f'{l_star = }')

def adaptive_controller_state(_t, x_state, u_input, params):
    """Internal state of adpative controller, f(t,x,u;p)"""
    
    # parameters
    gam_a = params["gam_a"]
    gam_b = params["gam_b"]
    Am = params["Am"]
    Bm = params["Bm"]
    signb = params["signb"]
    b0 = params["b0"]
    
    #print(gam1, gam2, Am, Bm)
    
    # controller inputs
    um = u_input[0]
    xp = u_input[1]
    xm = u_input[2]
    
    e1 = xm - xp

    # controller state
    Ad = x1 = x_state[0] # Ad
    Bd = x2 = x_state[1] # Bd
    
    # dynamics xd = f(x,u)
    d_x1 = - gam_a*e1*xp
    if (np.abs(Bd) > b0):
        k = (Am-Ad)/Bd
        l = Bm/Bd
        # control law
        up = k*xp + l*um
        d_x2 = - gam_b*e1*up
    elif (np.abs(Bd) == b0):
        k = (Am-Ad)/Bd
        l = Bm/Bd
        # control law
        up = k*xp + l*um
        if (gam_b*e1*up*signb >= 0):
            d_x2 = - gam_b*e1*up
        else:
            d_x2 = 0
    else:
        d_x2 = 0
    
    return [d_x1, d_x2]

def adaptive_controller_output(_t, x_state, u_input, params):
    """Algebraic output from adaptive controller, g(t,x,u;p)"""
    
    # parameters
    Am = params["Am"]
    Bm = params["Bm"]

    # controller inputs
    um = u_input[0]
    xp = u_input[1]
    xm = u_input[2]
    
    # controller state
    Ad = x_state[0]
    Bd = x_state[1]

    k = (Am-Ad)/Bd
    l = Bm/Bd

    # control law
    up = k*xp + l*um

    return [up, l, k, Ad, Bd]

b0=1e-3
params={"gam_a":1., "gam_b":1., "Am":Am, "Bm":Bm, "signb":np.sign(Bp), "b0":b0}

io_controller = ct.NonlinearIOSystem(
    adaptive_controller_state,
    adaptive_controller_output,
    inputs=3,
    outputs=('up', 'l', 'k', 'Ad', 'Bd'),
    states=2,
    params=params,
    name='control',
    dt=0
)

io_closed = ct.InterconnectedSystem(
    [io_plant, io_ref_model, io_controller],
    connections=[
        ['plant.up', 'control.up'],
        ['control.u[1]', 'plant.xp'],
        ['control.u[2]', 'ref_model.xm']
    ],
    inplist=['control.u[0]', 'ref_model.um'],
    outlist=['plant.xp', 'ref_model.xm', 'control.up', 'control.l', 'control.k', 'control.Ad', 'control.Bd'],
    dt=0
)

# set initial conditions
X0 = np.zeros((4, 1))
X0[0] = 0 # state of plant
X0[1] = 0 # state of ref_model
X0[2] = 0 # state of controller
X0[3] = b0*100 # state of controller


# set simulation duration and time steps
n_steps = 1000
Tend = 100

# define simulation time span 
t_vec = np.linspace(0, Tend, n_steps)
# define control input
uc_vec = np.zeros((2, n_steps))

rect = signal.square(2 * np.pi * 0.05 * t_vec)
sin = np.sin(2 * np.pi * 0.05 * t_vec) + np.sin(2 * np.pi * 0.5 * t_vec)
uc_vec[0, :] = rect
uc_vec[1, :] = uc_vec[0, :]

plt.figure(figsize=(16,8))
plt.plot(t_vec, uc_vec[0,:])
plt.title(r'Anregungssignal / Referenzsignal $u_m$')
plt.show()

# simulate the system, with different gammas

tout1, yout1 = ct.input_output_response(io_closed, t_vec, uc_vec, X0, params={"gam_a":0.2, "gam_b":0.2})
tout2, yout2 = ct.input_output_response(io_closed, t_vec, uc_vec, X0, params={"gam_a":1.0, "gam_b":1.0})
tout3, yout3 = ct.input_output_response(io_closed, t_vec, uc_vec, X0, params={"gam_a":5.0, "gam_b":5.0})

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.plot(tout1, yout1[0,:], label=r'$y_{\gamma = 0.2}$')
plt.plot(tout2, yout2[0,:], label=r'$y_{\gamma = 1.0}$')
plt.plot(tout2, yout2[0,:], label=r'$y_{\gamma = 5.0}$')
plt.plot(tout1, yout1[1,:] ,label=r'$y_{m}$', linestyle='--')
plt.legend()
plt.title('Systemantworten $y_p, (y_m)$')
plt.subplot(2,1,2)
plt.plot(tout1, yout1[2,:], label=r'$u$')
plt.plot(tout2, yout2[2,:], label=r'$u$')
plt.plot(tout3, yout3[2,:], label=r'$u$')
plt.legend(loc=4)
plt.title(r'Regler $u_p$')
plt.show()

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.plot(tout1, yout1[5,:], label=r'$\gamma = 0.2$')
plt.plot(tout2, yout2[5,:], label=r'$\gamma = 1.0$')
plt.plot(tout3, yout3[5,:], label=r'$\gamma = 5.0$')
plt.legend(loc=4)
plt.title(r'Systemparameter $\hat{A}$')
plt.subplot(2,1,2)
plt.plot(tout1, yout1[6,:], label=r'$\gamma = 0.2$')
plt.plot(tout2, yout2[6,:], label=r'$\gamma = 1.0$')
plt.plot(tout3, yout3[6,:], label=r'$\gamma = 5.0$')
plt.legend(loc=4)
plt.title(r'Systemparameter $\hat{B}$')
plt.show()

plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.plot(tout1, yout1[4,:], label=r'$\gamma = 0.2$')
plt.plot(tout2, yout2[4,:], label=r'$\gamma = 1.0$')
plt.plot(tout3, yout3[4,:], label=r'$\gamma = 5.0$')
plt.hlines(k_star, 0, Tend, label=r'$k^{\ast}$', color='black', linestyle='--')
plt.legend(loc=4)
plt.title(r'Reglerparameter $k$')
plt.subplot(2,1,2)
plt.plot(tout1, yout1[3,:], label=r'$\gamma = 0.2$')
plt.plot(tout2, yout2[3,:], label=r'$\gamma = 1.0$')
plt.plot(tout3, yout3[3,:], label=r'$\gamma = 5.0$')
plt.hlines(l_star, 0, Tend, label=r'$l^{\ast}$', color='black', linestyle='--')
plt.legend(loc=4)
plt.title(r'Reglerparameter $l$')
plt.show()
