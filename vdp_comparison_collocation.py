# Control of the Van der Pol
# oscillator using pure CasADi.

import casadi
import casadi.tools as ctools
import numpy as np
import matplotlib.pyplot as plt

import time


def get_collocation_points(d,include0=True,include1=False):
    tau_root = casadi.collocation_points(d, 'legendre')
    if include0:
        tau_root = [0] + tau_root
    if include1:
        tau_root = tau_root + [1]
    return tau_root


def get_collocation_weights(tau_root):
    d = len(tau_root)
    # Coefficients of the collocation equation
    C = np.zeros((d, d))

    # Coefficients of the continuity equation
    D = np.zeros(d)

    # Coefficients of the quadrature function
    B = np.zeros(d)

    # Construct polynomial basis
    for j in range(d):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points
        # to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)
    return [C,D,B]


def get_collocation_args(k,var,include0=True,include1=False):
    Nx = var['x'][k].shape[0]
    colloc_args = casadi.blocksplit(var['xc'][k], Nx, 1)[0]
    if include0:
        colloc_args = [var['x'][k]] + colloc_args
    if include1:
        colloc_args = colloc_args + [var['x'][k + 1]]
    return colloc_args

# Define model and get simulator.
Delta = .25
Nt = 10
Nx = 2
Nu = 1
Nc = 3 # Number of collocation points.
def ode(x,u):
    dxdt = [
        (1 - x[1]*x[1])*x[0] - x[1] + u,
        x[0]]
    return np.array(dxdt)

# Define CasADi symbolic variables.
x = casadi.SX.sym("x",Nx)
u = casadi.SX.sym("u",Nu)

# Make integrator object.
ode_integrator = dict(x=x,p=u,
    ode=ode(x,u))
intoptions = {
    "abstol" : 1e-8,
    "reltol" : 1e-8,
    "tf" : Delta,
}
vdp = casadi.integrator("int_ode",
    "cvodes", ode_integrator, intoptions)

# Then get nonlinear casadi functions.
ode_casadi = casadi.Function(
    "ode",[x,u],[ode(x,u)])

# Get collocation weights.
[C,D,B] = get_collocation_weights(get_collocation_points(Nc))

# Define stage cost and terminal weight.
lfunc = (casadi.mtimes(x.T, x)
    + casadi.mtimes(u.T, u))
l = casadi.Function("l", [x,u], [lfunc])

Pffunc = casadi.mtimes(x.T, x)
Pf = casadi.Function("Pf", [x], [Pffunc])

# Bounds on u.
uub = 1
ulb = -.75

# Make optimizers.
x0 = np.array([0,1])

# Create variables struct.
var = ctools.struct_symSX([(
    ctools.entry("x",shape=(Nx,),repeat=Nt+1),
    ctools.entry("xc", shape=(Nx,Nc), repeat=Nt),
    ctools.entry("u",shape=(Nu,),repeat=Nt),
)])
varlb = var(-np.inf)
varub = var(np.inf)
varguess = var(0)

# Adjust the relevant constraints.
for t in range(Nt):
    varlb["u",t,:] = ulb
    varub["u",t,:] = uub

# Now build up constraints and objective.
obj = casadi.SX(0)
con = []
for t in range(Nt):
    theseargs = get_collocation_args(t,var)
    # For all collocation points
    for j in range(1,Nc+1):
        xp_jt = 0
        for r in range(Nc+1):
            xp_jt += C[r,j] * theseargs[r]

        # Add collocation equations to the NLP
        fk = ode_casadi(theseargs[j], var["u", t])
        con.append(Delta * fk - xp_jt)

    # Get an expression for the state at the end of the finite element
    xf_k = 0
    for r in range(Nc+1):
        xf_k += casadi.mtimes(D[r], theseargs[r])

    # Add continuity equation to NLP
    con.append(var['x'][t + 1] - xf_k)

    obj += l(var["x",t], var["u",t])
obj += Pf(var["x",Nt])

# Build solver object.
con = casadi.vertcat(*con)
conlb = np.zeros(((Nx+Nc*Nx)*Nt,))
conub = np.zeros(((Nx+Nc*Nx)*Nt,))

nlp = dict(x=var, f=obj, g=con)
nlpoptions = {
    "ipopt" : {
        "print_level" : 0,
        "max_cpu_time" : 60,
        "linear_solver" : "ma27",
        "max_iter" : 100,
    },
    "print_time" : False,
    
}
solver = casadi.nlpsol("solver",
    "ipopt", nlp, nlpoptions)

# Now simulate.
Nsim = 40
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0
u = np.zeros((Nsim,Nu))
iter_time = np.zeros((Nsim,1))
for t in range(Nsim):
    t0 = time.time()
    # Fix initial state.    
    varlb["x",0,:] = x[t,:]
    varub["x",0,:] = x[t,:]
    varguess["x",0,:] = x[t,:]
    args = dict(x0=varguess,
                lbx=varlb,
                ubx=varub,
                lbg=conlb,
                ubg=conub)
    
    # Solve nlp.    
    sol = solver(**args)
    status = solver.stats()["return_status"]
    optvar = var(sol["x"])
        
    t1 = time.time()
    # Print stats.
    print "%d: %s in %.4f seconds" % (t,status, t1 - t0)
    u[t,:] = optvar["u",0,:]
    iter_time[t] = t1-t0   
    
    # Simulate.
    vdpargs = dict(x0=x[t,:],
                   p=u[t,:])
    out = vdp(**vdpargs)
    x[t+1,:] = np.array(
        out["xf"]).flatten()
    
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

# x plots.    
for i in range(Nx):
    ax = fig.add_subplot(numrows,
        numcols,numcols*(i+1) - 1)
    ax.plot(times,x[:,i],"-k",label="System")
    ax.set_xlabel("Time")
    ax.set_ylabel("State %d" % (i + 1))

ax = fig.add_subplot(numrows,numcols,4)
ax.plot(iter_time*1E3,".-k")
ax.set_xlabel("Iteration")
ax.set_ylabel("Execution time [ms]")
fig.tight_layout(pad=.5)
fig.show()

# Uncomment the following lines if you want the plot to block
# the Python interpreter and stay open.
# plt.show()
