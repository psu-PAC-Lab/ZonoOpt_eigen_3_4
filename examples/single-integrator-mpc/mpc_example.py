import numpy as np
import scipy.sparse as sp
import zonoopt as zono
import matplotlib.pyplot as plt
import time
import subprocess

def build_mpc(X0, U, A, B, N):

    """Build constrained zonotope representation of feasible space of MPC 
    optimization problem."""

    # dims
    nx = A.shape[1]
    nu = B.shape[1]

    # index tracking
    idx = 0
    idx_x = []
    idx_u = []

    # initial state
    Z = X0 # range of possible states
    idx_x.append([j for j in range(idx, idx+nx)])
    idx += nx

    for k in range(N):

        # control
        Z = zono.cartesian_product(Z, U)

        idx_u.append([j for j in range(idx, idx+nu)])
        idx += nu

        # save off indices of factors if k = 0
        if k == 0:
            ind_xi_0 = Z.get_nG()
        
        # dynamics
        nZ = Z.get_n()
        I = sp.eye(nZ)
        AB = sp.hstack((sp.csc_matrix((nx, nZ-(nx+nu))), A, B))
        H = sp.vstack((I, AB))
        Z = zono.affine_map(Z, H)

        idx_x.append([j for j in range(idx, idx+nx)])
        idx += nx

    return Z, idx_x, idx_u, ind_xi_0

# update IC
def update_IC(Z, x0, R):
    
        """Update initial condition of MPC problem."""
    
        # IC as point object
        X0 = zono.Point(x0)
    
        # generalized intersection
        Z_ic = zono.intersection(Z, X0, R)
    
        return Z_ic

# run simulation
def run_sim(Z_bl, R0, N, x_ref, solver):

    # number of sim time steps
    n_sim = x_ref.shape[1] - N

    # simulate
    x = np.zeros(nx) # init
    t_cum = 0.0
    t_max = 0.0
    k_cum = 0
    k_max = 0

    x_vec = x # init
    sol = None # init
    u_vec = [] # init 

    for k in range(n_sim):

        # update IC
        Z = update_IC(Z_bl, x, R0)

        # reference
        xr = x_ref[:,k:k+N+1]
        xur = np.zeros(Z.get_n()) # init

        for i in range(N+1):
            xur[idx_x[i]] = xr[:,i]

        # cost gradient
        q = Z.get_G().transpose().dot(P).dot(Z.get_c() - xur)

        # update solver, only q and b will have changed
        solver.update_q(q)
        solver.update_b(Z.get_b())

        # warm start
        if sol is not None:
            x_ws = np.zeros(Z.get_nG())
            x_ws[:(len(x_ws)-ind_xi_0)] = sol.x[ind_xi_0:]

            u_ws = np.zeros(Z.get_nG())
            u_ws[:(len(u_ws)-ind_xi_0)] = sol.u[ind_xi_0:]

            solver.warmstart(x_ws, u_ws)

        # solve
        sol = solver.solve()
        xopt = Z.get_G().dot(sol.z) + Z.get_c()

        # extract control
        u = xopt[idx_u[0]]
        u_vec.append(u)

        # step dynamics
        x = A.dot(x) + B.dot(u)

        # logging
        t_cum += sol.run_time
        t_max = max(t_max, sol.run_time)
        k_cum += sol.k
        k_max = max(k_max, sol.k)
        
        x_vec = np.vstack((x_vec, x))

    # average solution time
    t_avg = t_cum/n_sim

    # average number of iterations
    k_avg = int(k_cum/n_sim)

    return x_vec, u_vec, t_avg, t_max, k_avg, k_max

def is_latex_installed():
    try:
        subprocess.run(["latex", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False



### PROBLEM SETUP ###

# time step
dt = 1.0

# 2D double integrator dynamics
# x = [x, y, xdot, y_dot]
# u = [x_ddot, y_ddot]
A = sp.csc_matrix(np.array(
             [[1., 0.],
              [0., 1.]]))
B = sp.csc_matrix(np.array(
                [[dt, 0.],
                [0., dt]]))
nx = 2
nu = 2

# state feasible set

# velocity constraints
v_max = 0.5
U = zono.make_regular_zono_2D(v_max, 8)

# MPC horizon
N = 40

# build feasible space of MPC problem
X0 = zono.Zono(sp.diags([1000., 1000.]), np.zeros(2))
Z_bl, idx_x, idx_u, ind_xi_0 = build_mpc(X0, U, A, B, N)

# intersect with dummy initial condition to get problem structure
R0 = sp.hstack((sp.eye(nx), sp.csc_matrix((nx, Z_bl.get_n()-nx)))) # generalized intersection over IC
Z = update_IC(Z_bl, np.zeros(nx), R0)

# MPC cost matrices
Q = sp.eye(nx)
R = sp.eye(nu)

P = sp.csc_matrix((0,0))
for i in range(N):
    P = sp.block_diag((P, Q, R))
P = sp.block_diag((P, Q))

# solver settings
settings = zono.ADMM_settings()
settings.eps_prim = 1e-2
settings.eps_dual = 1e-2
settings.k_max = 1000

# need to instantiate ADMM solver in order to take advantage of warm start and caching factorization
solver = zono.ADMM_solver()
if Z.is_0_1_form():
    xi_lb = np.zeros(Z.get_nG())
else:
    xi_lb = -np.ones(Z.get_nG())
xi_ub = np.ones(Z.get_nG())

P_zono = Z.get_G().transpose().dot(P).dot(Z.get_G())
q_zono = Z.get_G().transpose().dot(P).dot(Z.get_c())

solver.setup(P_zono, q_zono, Z.get_A(), Z.get_b(), xi_lb, xi_ub, settings)

t1 = time.time()
solver.factorize() # pre-factorization of matrices
tf = time.time()
print('Factorization time: ', tf-t1)
print(f'Z.get_n(): {Z.get_n()}, Z.get_nG(): {Z.get_nG()}, Z.get_nC(): {Z.get_nC()}')

### SIMULATION ###

# reference state
n_sim = 100
yp_ref = np.linspace(0, 50, n_sim+N)
xp_ref = 5*np.sin(0.3*yp_ref)
x_ref = np.vstack((xp_ref, yp_ref, np.zeros((nx-2, n_sim+N))))

# run sim for different max number of iterations
k_max = [1, 2, 3, 4, 5, 5000]
x_arr = []
u_arr = []
t_avg_arr = []
t_max_arr = []
k_avg_arr = []
k_max_arr = []

for k in k_max:
    
    # update max iterations
    settings.k_max = k
    solver.update_settings(settings)

    # run sim
    x, u, t_avg, t_max, k_avg, k_max_sim = run_sim(Z_bl, R0, N, x_ref, solver)

    # save off data
    x_arr.append(x)
    u_arr.append(u)
    t_avg_arr.append(t_avg)
    t_max_arr.append(t_max)
    k_avg_arr.append(k_avg)
    k_max_arr.append(k_max_sim)

### PLOT ###

# print solution times
for i in range(len(k_max)):
    print(f'k_max = {k_max[i]}, t_avg = {t_avg_arr[i]}, t_max = {t_max_arr[i]}, k_avg = {k_avg_arr[i]}, k_max = {k_max_arr[i]}')

# set up LaTeX rendering
if is_latex_installed():
    rc_context = {
        "text.usetex": True,
        "font.size": 10,
        "font.family": "serif",  # Choose a serif font like 'Times New Roman' or 'Computer Modern'
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
    }
else:
    print("LaTeX not installed, using default font.")
    rc_context = {
        "font.size": 10,
    }

inches_per_pt = 1 / 72.27
figsize = (245.71 * inches_per_pt, 0.9*245.71 * inches_per_pt)  # Convert pt to inches

# plot results
with plt.rc_context(rc_context):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # reference and simulation
    ax.plot(xp_ref[:n_sim], yp_ref[:n_sim], 'k--')
    for i in range(len(k_max)):
        ax.plot(x_arr[i][:,0], x_arr[i][:,1])

    # legend
    legend = [r'$\mathbf{x}_r$']
    for i in range(len(k_max)):
        k = k_max[i]
        if k == 5000:
            legend.append(fr'$\bar{{k}} = {k_avg_arr[i]}$')
        else:
            legend.append(fr'$k = {k}$')
    ax.legend(legend)

    # axis lables
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.axis('equal')
    ax.set_ybound(lower=-5, upper=40)
    ax.set_xbound(lower=-10, upper=20)
    

    # save figure
    if is_latex_installed():
        plt.savefig('mpc_traj.pgf')

    plt.show()