# Control of the Van der Pol
# oscillator using pure CasADi.

import casadi
import casadi.tools as ctools
import numpy as np
import matplotlib.pyplot as plt

import time
import scipy.linalg

def c2d(A, B, Delta, Bp=None, f=None, asdict=False):
    """
    Discretizes affine system (A, B, Bp, f) with timestep Delta.
    This includes disturbances and a potentially nonzero steady-state, although
    Bp and f can be omitted if they are not present.
    If asdict=True, return value will be a dictionary with entries A, B, Bp,
    and f. Otherwise, the return value will be a 4-element list [A, B, Bp, f]
    if Bp and f are provided, otherwise a 2-element list [A, B].
    """
    n = A.shape[0]
    I = np.eye(n)
    D = scipy.linalg.expm(Delta * np.vstack((np.hstack([A, I]),
                                             np.zeros((n, 2 * n)))))
    Ad = D[:n, :n]
    Id = D[:n, n:]
    Bd = Id.dot(B)
    Bpd = None if Bp is None else Id.dot(Bp)
    fd = None if f is None else Id.dot(f)

    if asdict:
        retval = dict(A=Ad, B=Bd, Bp=Bpd, f=fd)
    elif Bp is None and f is None:
        retval = [Ad, Bd]
    else:
        retval = [Ad, Bd, Bpd, fd]
    return retval

def _calc_lin_disc_wrapper_for_mp_map(item):
    """ Function wrapper for map or multiprocessing.map . """
    _fi, _xi, _ui, _Delta = item
    Ai = _fi.jacobian(0, 0)(_xi, _ui)[0].full()
    Bi = _fi.jacobian(1, 0)(_xi, _ui)[0].full()
    # Gi = _fi.jacobian(2, 0)(_xi, _ui)[0].full()
    Ei = _fi(_xi, _ui).full().ravel() - Ai.dot(_xi).ravel() - Bi.dot(_ui).ravel()  # - Gi.dot(_wi).ravel()
    # [Ai[:], Bi[:], Gi[:], Ei[:]] = c2d(A=Ai, B=Bi, Delta=_Delta, Bp=Gi, f=Ei)
    [Ai[:], Bi[:], _, Ei[:]] = c2d(A=Ai, B=Bi, Delta=_Delta, f=Ei)
    return Ai, Bi, Ei


isQP = True
updateLTV = False

# Define model and get simulator.
Delta = .25
Nt = 10
Nx = 2
Nu = 1
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
u0 = np.array([0])

# Create variables struct.
var = ctools.struct_symSX([(
    ctools.entry("x",shape=(Nx,),repeat=Nt+1),
    ctools.entry("u",shape=(Nu,),repeat=Nt),
)])
varlb = var(-np.inf)
varub = var(np.inf)
varguess = var(0)

# Create parameters struct.
par = ctools.struct_symSX([
    ctools.entry("Ad", repeat=Nt, shape=(Nx, Nx)),
    ctools.entry("Bd", repeat=Nt, shape=(Nx, Nu)),
    ctools.entry("fd", repeat=Nt, shape=(Nx, 1))])
parguess = par(0)

# Adjust the relevant constraints.
for t in range(Nt):
    varlb["u",t,:] = ulb
    varub["u",t,:] = uub

# Set initial values to the LTV system parameters.
[A0, B0, f0] = _calc_lin_disc_wrapper_for_mp_map([ode_casadi,x0,u0,Delta])
for t in range(Nt):
    [parguess['Ad',t], parguess['Bd',t], parguess['fd',t]] = \
        _calc_lin_disc_wrapper_for_mp_map([ode_casadi,x0,u0,Delta])

# Now build up constraints and objective.
obj = casadi.SX(0)
con = []
for t in range(Nt):
    con.append(var["x", t + 1] -
               casadi.mtimes(par["Ad", t], var["x", t]) -
               casadi.mtimes(par["Bd", t], var["u", t]) -
               par["fd", t])
    obj += l(var["x",t], var["u",t])
obj += Pf(var["x",Nt])

# Build solver object.
con = casadi.vertcat(*con)
conlb = np.zeros((Nx*Nt,))
conub = np.zeros((Nx*Nt,))

if isQP:
    qp = dict(x=var, f=obj, g=con, p=par)
    qpoptions = {
        "printLevel": 'none',
        "print_time": False,
        'sparse': True
    }
    solver = casadi.qpsol('solver', 'qpoases', qp, qpoptions)
else:
    nlp = dict(x=var, f=obj, g=con, p=par)
    nlpoptions = {
        "ipopt" : {
            "print_level" : 0,
            "max_cpu_time" : 60,
            "linear_solver" : "ma27",
            "max_iter" : 100,
            "jac_c_constant": "yes",
            "jac_d_constant": "yes",
            "hessian_constant": "yes",
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
                p=parguess,
                lbx=varlb,
                ubx=varub,
                lbg=conlb,
                ubg=conub)
    
    # Solve nlp.    
    sol = solver(**args)
    if isQP:
        # We have no stats from qpsol.
        status = "Solve_Succeeded"
    else:
        status = solver.stats()["return_status"]
    optvar = var(sol["x"])

    # Update LTV system parameters before next iteration

    if updateLTV or t==0:
        # Update all the parameters of the LTV system.
        parguess["Ad", :],\
        parguess["Bd", :],\
        parguess["fd", :] = zip(*map(_calc_lin_disc_wrapper_for_mp_map,
                                     zip([ode_casadi for _k in xrange(Nt)],
                                         optvar["x",:-1],
                                         optvar["u"],
                                         [Delta for _k in xrange(Nt)])))
    else:
        # Shift previous parameters and update the last one.
        parguess["Ad",0:-1] = parguess["Ad",1:]
        parguess["Bd",0:-1] = parguess["Bd",1:]
        parguess["fd",0:-1] = parguess["fd",1:]
        parguess["Ad",-1], parguess["Bd",-1], parguess["fd",-1] = \
            _calc_lin_disc_wrapper_for_mp_map([ode_casadi,optvar["x",-2],optvar["u",-1], Delta])


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
