# Control of the Van der Pol
# oscillator using CasADi Opti stack.

import casadi
import numpy as np
import matplotlib.pyplot as plt

import time

# Set to True if you want to get plots at the end of the simulation.
plot_results = True

# Define model and get simulator.
Delta = .25
Nt = 10
Nx = 2
Nu = 1

def ode(x, u):
    dxdt = [
        (1 - x[1]*x[1])*x[0] - x[1] + u,
        x[0]]
    return np.array(dxdt)

# Define CasADi symbolic variables.
x = casadi.SX.sym("x", Nx)
u = casadi.SX.sym("u", Nu)

# Make integrator object.
ode_integrator = dict(x=x, p=u, ode=ode(x, u))
intoptions = {
    "abstol": 1e-8,
    "reltol": 1e-8,
}
vdp = casadi.integrator("int_ode", "cvodes", ode_integrator, 0, Delta, intoptions)

# Then get nonlinear casadi functions and RK4 discretization.
ode_casadi = casadi.Function("ode", [x, u], [ode(x, u)])

k1 = ode_casadi(x, u)
k2 = ode_casadi(x + Delta/2*k1, u)
k3 = ode_casadi(x + Delta/2*k2, u)
k4 = ode_casadi(x + Delta*k3, u)
xrk4 = x + Delta/6*(k1 + 2*k2 + 2*k3 + k4)    
ode_rk4_casadi = casadi.Function("ode_rk4", [x, u], [xrk4])

# Define stage cost and terminal cost.
lfunc = (casadi.mtimes(x.T, x) + casadi.mtimes(u.T, u))
l = casadi.Function("l", [x, u], [lfunc])

Pffunc = casadi.mtimes(x.T, x)
Pf = casadi.Function("Pf", [x], [Pffunc])

# Bounds on u.
uub = 1
ulb = -.75

# Initial state
x0 = np.array([0, 1])

# Create Opti instance
opti = casadi.Opti()

# Decision variables
x_opti = opti.variable(Nx, Nt+1)   # States
u_opti = opti.variable(Nu, Nt)     # Controls
x0_par = opti.parameter(Nx, 1)
    
# Objective function
obj = 0
for k in range(Nt):
    obj += l(x_opti[:, k], u_opti[:, k])
obj += Pf(x_opti[:, Nt])
opti.minimize(obj)

# Constraints
for k in range(Nt):
    # Dynamics constraint
    opti.subject_to(x_opti[:, k+1] == ode_rk4_casadi(x_opti[:, k], u_opti[:, k]))
    # Control bounds
    opti.subject_to(opti.bounded(ulb, u_opti[:, k], uub))
# Initial state constraint
opti.subject_to(x_opti[:, 0] == x0_par)
    
# Solver options
opts = {
    "ipopt.print_level": 0,
    "ipopt.max_cpu_time": 60,
    "ipopt.max_iter": 100,
    "print_time": False
}

# Solve NLP
opti.solver('ipopt', opts)

# Now simulate.
Nsim = 40
times = Delta*Nsim*np.linspace(0, 1, Nsim+1)
x = np.zeros((Nsim+1, Nx))
x[0, :] = x0
u = np.zeros((Nsim, Nu))
iter_time = np.zeros((Nsim, 1))

for t in range(Nsim):
    t0 = time.time()
    opti.set_value(x0_par, x[t, :])
    sol = opti.solve()
    t1 = time.time()
    status = opti.stats()["return_status"]
    
    # Print stats
    print ("%d: %s in %.4f seconds" % (t,status, t1 - t0))

    # Store solution
    u[t, :] = sol.value(u_opti[:, 0])
    # Store elapsed time
    iter_time[t] = t1 - t0
    
    # Simulate
    vdpargs = dict(x0=x[t, :], p=u[t, :])
    out = vdp(**vdpargs)
    x[t+1, :] = np.array(out["xf"]).flatten()

if plot_results:
    # Plots.
    fig = plt.figure()
    numrows = max(Nx,Nu)
    numcols = 2

    # u plots. Need to repeat last element
    # for stairstep plot.
    u = np.concatenate((u,u[-1:,:]))
    for i in range(Nu):
        ax = fig.add_subplot(numrows,
            numcols,numcols*(i+1))
        ax.step(times,u[:,i],"-k",where="post")
        ax.set_xlabel("Time")
        ax.set_ylabel("Control %d" % (i + 1))
        ax.grid()

    # x plots.    
    for i in range(Nx):
        ax = fig.add_subplot(numrows,
            numcols,numcols*(i+1) - 1)
        ax.plot(times,x[:,i],"-k",label="System")
        ax.set_xlabel("Time")
        ax.set_ylabel("State %d" % (i + 1))
        ax.grid()

    ax = fig.add_subplot(numrows,numcols,4)
    ax.plot(iter_time*1E3,".-k")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Execution time [ms]")
    ax.grid()
    fig.tight_layout(pad=.5)
    plt.show()
