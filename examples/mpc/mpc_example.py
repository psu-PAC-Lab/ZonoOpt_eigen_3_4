import zonoopt as zono
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import subprocess

# solver options: 'zonoopt', 'osqp', 'gurobi'
solver_flag = 'zonoopt'

# representation options
use_hrep = False

# sampling fineness factor
sample_factor = 1

# flag to iteratively build and solve MPC problem for different methods
do_method_comparison = False

if solver_flag=='osqp' or do_method_comparison:
    try:
        import osqp # used version 0.6.7.post3 in paper
        
    except ImportError:
        print('OSQP not found')
        exit()

if solver_flag=='gurobi' or do_method_comparison:
    try:
        import gurobipy as gp # used version 12.0.1 in paper
    except ImportError:
        print('Gurobi not found')
        exit()

if use_hrep or do_method_comparison:
    try:
        import shapely
    except ImportError:
        print('Shapely not found')
        exit()

if do_method_comparison:
    try:
        import pandas as pd
    except ImportError:
        print('Pandas not found')
        exit()

class HrepPoly():
    def __init__(self, A=None, b=None):
        self.A = A
        self.b = b

def vrep_2_hrep(V):
    """Convert V-rep polytope to H-rep polytope"""
    
    # make shapely polygon
    P = shapely.geometry.Polygon(V)

    # make sure vertices oriented counter-clockwise
    if not P.exterior.is_ccw:
        P = shapely.geometry.polygon.orient(P, sign=1.0)

    # get vertices
    V = np.array(P.exterior.coords.xy).transpose()
    V = V[0:-1,:] # remove duplicate vertex

    # get H-rep
    nV = V.shape[0] # number of vertices
    A = np.zeros((nV, 2)) # init
    b = np.zeros(nV) # init
    for i in range(nV):
        xi, yi = V[i]
        xip1, yip1 = V[(i+1) % nV]
        dx = xip1 - xi
        dy = yip1 - yi
        A[i] = [dy, -dx]
        b[i] = dy*xi - dx*yi
        Ai_norm = np.linalg.norm(A[i])
        if Ai_norm > 1e-15:
            A[i] /= Ai_norm
            b[i] /= Ai_norm

    return HrepPoly(A, b)

def is_latex_installed():
    try:
        subprocess.run(["latex", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
                

def zono_mpc(x0, xr, A, B, Q, R, N, X_arr, U, settings=zono.OptSettings(), sol=zono.OptSolution()):
    """Build and solve MPC problem using zonoopt set operations.
    x0 = initial state
    xr = reference trajectory (N+1 x n numpy array)
    A = state transition matrix
    B = control matrix
    Q = state cost matrix
    R = control cost matrix
    N = prediction horizon
    X_arr = array of state constraints
    U = control constraint set
    settings = zonoopt.OptSettings object
    sol = zonoopt.OptSolution object
    """

    # dims
    nx = A.shape[1]
    nu = B.shape[1]

    # index tracking
    idx = 0
    idx_x = []
    idx_u = []

    # initial state
    Z = zono.Point(x0)
    idx_x.append([j for j in range(idx, idx+nx)])
    idx += nx

    # init cost
    P = Q
    q = np.zeros(nx)

    for k in range(N):

        # control
        Z = zono.cartesian_product(Z, U)

        idx_u.append([j for j in range(idx, idx+nu)])
        idx += nu
        
        # dynamics
        nZ = Z.get_n()
        Z = zono.cartesian_product(Z, X_arr[k+1])
        ABmI = sp.hstack((sp.csc_matrix((nx, nZ-(nx+nu))), A, B, -sp.eye(nx)))
        Z = zono.intersection(Z, zono.Point(np.zeros(nx)), ABmI)

        idx_x.append([j for j in range(idx, idx+nx)])
        idx += nx

        # cost
        P = sp.block_diag((P, R, Q))
        q = np.hstack([q, np.zeros(nu), -Q.dot(xr[k])])

    if solver_flag=='zonoopt':

        # optimize
        xopt = Z.optimize_over(P, q, settings=settings, solution=sol)

    elif solver_flag=='osqp':
        
        # build osqp problem
        P_tilde = Z.get_G().transpose().dot(P.dot(Z.get_G()))
        q_tilde = Z.get_G().transpose().dot(P*Z.get_c() + q)
        A_osqp = Z.get_A()
        A_osqp = sp.vstack([A_osqp, sp.eye(Z.get_nG())])
        if Z.is_0_1_form():
            l_osqp = np.hstack([Z.get_b(), np.zeros(Z.get_nG())])
        else:
            l_osqp = np.hstack([Z.get_b(), -1*np.ones(Z.get_nG())])
        u_osqp = np.hstack([Z.get_b(), np.ones(Z.get_nG())])

        # create object
        prob = osqp.OSQP()
        prob.setup(P=P_tilde, q=q_tilde, A=A_osqp, l=l_osqp, u=u_osqp, eps_rel=0.0, eps_abs=settings.eps_prim)

        # solve
        osqp_sol = prob.solve()
        xopt = Z.get_G()*osqp_sol.x + Z.get_c()
        
        # logging
        sol.run_time = osqp_sol.info.run_time
        sol.iter = osqp_sol.info.iter
        sol.startup_time = osqp_sol.info.setup_time
        sol.infeasible = osqp_sol.info.status_val != 1

    elif solver_flag=='gurobi':
        
        # build gurobi model
        P_tilde = Z.get_G().transpose().dot(P.dot(Z.get_G()))
        q_tilde = Z.get_G().transpose().dot(P*Z.get_c() + q)

        # create model
        prob = gp.Model()

        # add variables
        if Z.is_0_1_form():
            x_gurobi = prob.addMVar(Z.get_nG(), lb=np.zeros(Z.get_nG()), ub=np.ones(Z.get_nG()))
        else:
            x_gurobi = prob.addMVar(Z.get_nG(), lb=-np.ones(Z.get_nG()), ub=np.ones(Z.get_nG()))

        # add constraints
        A_gurobi = Z.get_A()
        b_gurobi = Z.get_b()
        prob.addConstr(A_gurobi.dot(x_gurobi) == b_gurobi)

        # add objective
        prob.setMObjective(P_tilde, q_tilde, 0.0, sense=gp.GRB.MINIMIZE)

        # optimize
        prob.optimize()
        xi_opt = np.array([x.X for x in prob.getVars()])
        xopt = Z.get_G()*xi_opt + Z.get_c()

        # logging
        sol.run_time = prob.Runtime
        sol.iter = prob.BarIterCount
        sol.startup_time = 0.0
        sol.infeasible = prob.Status != 2

    else:
        return ValueError('Unknown solver flag')

    # state, input trajectory
    x_traj = []
    u_traj = []
    for idx in idx_x:
        x_traj.append(xopt[idx])
    for idx in idx_u:
        u_traj.append(xopt[idx])

    # return
    return x_traj, u_traj, Z

def hrep_mpc(x0, xr, A, B, Q, R, N, X_arr, U, settings=zono.OptSettings(), sol=zono.OptSolution()):
    """Build and solve MPC problem in H-rep form.
    x0 = initial state
    xr = reference trajectory (N+1 x n numpy array)
    A = state transition matrix
    B = control matrix
    Q = state cost matrix
    R = control cost matrix
    N = prediction horizon
    X_arr = array of Hrep polytopes
    U = control constraint set as a polytope
    settings = zonoopt.OptSettings object
    sol = zonoopt.OptSolution object
    """

    # problem dimensions
    nx = A.shape[1]
    nu = B.shape[1]

    # index tracking
    idx = 0
    idx_x = []
    idx_u = []

    # build equality constraint matrix
    C = -sp.eye(nx)
    idx_x.append([j for j in range(idx, idx+nx)])
    idx += nx

    for k in range(N):
        C = sp.hstack((C, sp.csc_matrix((C.shape[0], nu))))
        AB = sp.hstack((sp.csc_matrix((nx, C.shape[1]-nx-nu)), sp.hstack((A, B))))
        C = sp.vstack((C, AB))

        idx_u.append([j for j in range(idx, idx+nu)])
        idx += nu

        mI = sp.vstack((sp.csc_matrix((C.shape[0]-nx, nx)), -sp.eye(nx)))
        C = sp.hstack((C, mI))

        idx_x.append([j for j in range(idx, idx+nx)])
        idx += nx

    # equality constraint vector
    d = np.zeros(C.shape[0])
    d[0:nx] = -x0

    # build inequality constraint matrix
    G = sp.hstack((sp.csc_matrix((U.A.shape[0],nx)), U.A))
    for k in range(1,N+1):
        if k < N:
            G = sp.block_diag((G, X_arr[k].A, U.A))
        else:
            G = sp.block_diag((G, X_arr[k].A))

    # inequality constraint vector
    w = U.b
    for k in range(1, N+1):
        if k < N:
            w = np.hstack((w, X_arr[k].b, U.b))
        else:
            w = np.hstack((w, X_arr[k].b))

    # cost function
    P = sp.csc_matrix((0,0))
    for k in range(N+1):
        if k < N:
            P = sp.block_diag((P, Q, R))
        else:
            P = sp.block_diag((P, Q))
    
    q = np.zeros(nx+nu)
    for k in range(1,N+1):
        if k < N:
            q = np.hstack((q, -Q.dot(xr[k-1]), np.zeros(nu)))
        else:
            q = np.hstack((q, -Q.dot(xr[k-1])))

    # ensure all matrices are in CSC format
    C = C.tocsc()
    G = G.tocsc()
    P = P.tocsc()

    # solve
    if solver_flag=='zonoopt':
        raise ValueError('Not implemented')

    elif solver_flag=='osqp':

        # build osqp problem
        A_osqp = C
        A_osqp = sp.vstack([A_osqp, G])
        l_osqp = np.hstack([d, -np.inf*np.ones(len(w))])
        u_osqp = np.hstack([d, w])

        # create object
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A_osqp, l=l_osqp, u=u_osqp, eps_rel=0.0, eps_abs=settings.eps_prim)

        # solve
        osqp_sol = prob.solve()
        xopt = osqp_sol.x
        
        # logging
        sol.run_time = osqp_sol.info.run_time
        sol.iter = osqp_sol.info.iter
        sol.startup_time = osqp_sol.info.setup_time
        sol.infeasible = osqp_sol.info.status_val != 1

    elif solver_flag=='gurobi':

        # create model
        prob = gp.Model()

        # add variables
        n_vars = len(q)
        x_gurobi = prob.addMVar(n_vars, lb=-gp.GRB.INFINITY*np.ones(n_vars), ub=gp.GRB.INFINITY*np.ones(n_vars))

        # add constraints
        prob.addConstr(C.dot(x_gurobi) == d)
        prob.addConstr(G.dot(x_gurobi) <= w)

        # add objective
        prob.setMObjective(P, q, 0.0, sense=gp.GRB.MINIMIZE)

        # optimize
        prob.optimize()
        xopt = np.array([x.X for x in prob.getVars()])

        # logging
        sol.run_time = prob.Runtime
        sol.iter = prob.BarIterCount
        sol.startup_time = 0.0
        sol.infeasible = prob.Status != 2

    else:
        raise ValueError('Unknown solver flag')
    
    # return state, input trajectory
    x_traj = []
    u_traj = []
    for idx in idx_x:
        x_traj.append(xopt[idx])
    for idx in idx_u:
        u_traj.append(xopt[idx])
    
    # return control input
    return x_traj, u_traj, P, C, G

def solve_example(make_plot=False):

    ### Environment setup ###

    # path plan
    path_plan = np.array([[0.0, -10.0],
                        [0.0, 15.0],
                        [15.0, 20.0],
                        [15.0, 35.0]])

    # get points at uniform distance traveled along path
    d_path = np.hstack([0.0, np.cumsum(np.sqrt(np.sum(np.diff(path_plan, axis=0)**2, axis=1)))])
    ds = 1.0/sample_factor
    d_vec = np.linspace(0.0, d_path[-1], int(d_path[-1]/ds))
    path_x = np.interp(d_vec, d_path, path_plan[:, 0])
    path_y = np.interp(d_vec, d_path, path_plan[:, 1])

    # get indices for the part of the path in the middle
    idx_middle = np.where((path_y >= 15.0) & (path_y <= 20.0))[0]


    # obstacles
    m = (20.0-15.0)/(15.0-0.0) # slope
    V_O1 = np.array([[5.0, 15.6],
                [10.0, 15.6 + 5.0*m],
                [4.0, 10.6],
                [15.0, 12.1],
                [13.0, 6.6],
                [7.5, 8.1],
                [7.2, 16.8]])
    O1 = zono.vrep_2_conzono(V_O1)

    V_02 = np.array([[4.0, 17.6],
                    [9.0, 17.6 + 5.0*m],
                    [1.0, 21.4],
                    [2.0, 26.4],
                    [6.0, 30.4],
                    [11.0, 26.4],
                    [6.9, 17.9]])
    O2 = zono.vrep_2_conzono(V_02)

    # base zonotope for constraint sets
    Z_base = zono.make_regular_zono_2D(1., 6)

    # variable constraint sets
    len_front = int(len(idx_middle)/2)
    len_back = len(idx_middle) - len_front
    rt_vec_middle = np.hstack([np.linspace(2.0, 0.4, len_front), 
                        np.linspace(0.4, 2.0, len_back)])
    rs_vec_middle = np.hstack([np.linspace(2.0, 2.0, len_front), 
                        np.linspace(2.0, 2.0, len_back)])

    rt_vec = 4.5*np.ones(len(d_vec))
    rs_vec = 4.5*np.ones(len(d_vec))
    rt_vec[idx_middle] = rt_vec_middle
    rs_vec[idx_middle] = rs_vec_middle

    phi_vec = np.hstack(np.arctan2(np.diff(path_y), np.diff(path_x)))
    phi_vec = np.hstack([phi_vec[0], phi_vec])

    Z_cons_arr = []
    for i in range(len(d_vec)):
        C_rot = sp.csc_matrix([[np.cos(phi_vec[i]), -np.sin(phi_vec[i])],
                                [np.sin(phi_vec[i]), np.cos(phi_vec[i])]])
        C_lin = sp.diags([rs_vec[i], rt_vec[i]])
        C = C_rot*C_lin
        b = np.array([path_x[i], path_y[i]])

        Z_cons_arr.append(zono.affine_map(Z_base, C, b))


    ### MPC setup ###

    # time step
    dt = 1.0/sample_factor

    # 2D double integrator dynamics
    # x = [x, y, xdot, y_dot]
    # u = [x_ddot, y_ddot]
    A = sp.csc_matrix(np.array(
                [[1., 0., dt, 0.],
                [0., 1., 0., dt],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]]))
    B = sp.csc_matrix(np.array(
                [[0.5*dt**2, 0.],
                    [0., 0.5*dt**2],
                    [dt, 0.],
                    [0., dt]]))

    # state feasible set

    # velocity constraints
    v_max = 5.0
    Xv = zono.make_regular_zono_2D(v_max, 12)

    # input feasible set

    # turn rate constraints
    om_max = 75*np.pi/180
    v_min = 0.1 # fictitious, om_max not enforced below this
    U = zono.make_regular_zono_2D(om_max*v_min, 12)

    # MPC horizon
    N = 55*sample_factor

    # cost function matrices
    Q = sp.diags([1.0, 1.0, 0.0, 0.0])
    R = 10*sp.eye(2)

    # solver settings
    settings = zono.OptSettings()
    settings.inf_norm_conv = True

    ### Solve example ###

    # sim length
    n_sim = len(d_vec)

    # reference
    x_ref = np.zeros((0, 4)) # init
    for i in range(n_sim):
        x_ref = np.vstack((x_ref, np.array([path_x[i], path_y[i], 0.0, 0.0])))
    for k in range(N+1):
        x_ref = np.vstack((x_ref, x_ref[-1,:]))
    x_ref = x_ref.transpose()

    # initial condition
    x = np.array([0.0, -10.0, 0.0, 0.0]) # init

    # reference
    xr = []
    for i in range(1,N+1):
        xr.append(x_ref[:,i].flatten())

    # build constraint sets
    X_arr = []
    for i in range(N+1):
        if i >= len(Z_cons_arr):
            X_arr.append(zono.cartesian_product(Z_cons_arr[-1], Xv))
        else:
            X_arr.append(zono.cartesian_product(Z_cons_arr[i], Xv))

    # build and solve MPC
    sol = zono.OptSolution()
    if use_hrep:
        
        # build constraint sets in H-rep
        X_arr_hrep = []
        for i in range(N+1):
            Xv_hrep = vrep_2_hrep(zono.get_vertices(Xv))
            if i >= len(Z_cons_arr):
                Xp_hrep = vrep_2_hrep(zono.get_vertices(Z_cons_arr[-1]))
            else:
                Xp_hrep = vrep_2_hrep(zono.get_vertices(Z_cons_arr[i]))
            A_hrep = sp.block_diag((sp.csc_matrix(Xp_hrep.A), sp.csc_matrix(Xv_hrep.A)))
            b_hrep = np.hstack((Xp_hrep.b, Xv_hrep.b))
            X_arr_hrep.append(HrepPoly(A_hrep, b_hrep))

        U_hrep = vrep_2_hrep(zono.get_vertices(U))

        # solve
        x_traj, u_traj, Ph, Ch, Gh = hrep_mpc(x, xr, A, B, Q, R, N, X_arr_hrep, U_hrep, settings=settings, sol=sol)

    else:
        x_traj, u_traj, Z_mpc = zono_mpc(x, xr, A, B, Q, R, N, X_arr, U, settings=settings, sol=sol)


    # display
    print(f'iter: {sol.iter}, run time: {sol.run_time}, startup time = {sol.startup_time}')

    ### make plot ###
    if make_plot:

        # display complexity of set representation
        if use_hrep:
            print(f'P dims = {Ph.shape}, C dims = {Ch.shape}, G dims = {Gh.shape}')
            print(f'NNZ(P): {Ph.nnz}, NNZ(C): {Ch.nnz}, NNZ(G): {Gh.nnz}')
        else:
            print(f'Z.n: {Z_mpc.get_n()}, Z.nG: {Z_mpc.get_nG()}, Z.nC: {Z_mpc.get_nC()}')
            print(f'NNZ(G): {Z_mpc.get_G().nnz}, NNZ(A): {Z_mpc.get_A().nnz}')


        textwidth_pt = 10.
        if is_latex_installed():
            rc_context = {
                "text.usetex": True,
                "font.size": textwidth_pt,
                "font.family": "serif",  # Choose a serif font like 'Times New Roman' or 'Computer Modern'
                "pgf.texsystem": "pdflatex",
                "pgf.rcfonts": False,
            }
        else:
            print("LaTeX not installed, using default font.")
            rc_context = {
                "font.size": textwidth_pt,
            }

        inches_per_pt = 1 / 72.27
        figsize = (245.71 * inches_per_pt, 0.7*245.71 * inches_per_pt)  # Convert pt to inches

        # plot
        with plt.rc_context(rc_context):

            # flip x and y axes
            C_flip = sp.csc_matrix([[0, 1],
                                [1, 0]])

            # figure
            fig = plt.figure(constrained_layout=True, figsize=figsize)
            ax = fig.add_subplot(111)

            # path plan
            ax.plot(path_plan[:, 1], path_plan[:, 0], '-k')

            # obstacles
            zono.plot(zono.affine_map(O1, C_flip), ax=ax, color='g', alpha=0.7, edgecolor='k')
            zono.plot(zono.affine_map(O2, C_flip), ax=ax, color='g', alpha=0.7, edgecolor='k')

            # time-varying state constraints (spatio-temporal corridor)
            for k in range(n_sim):
                zono.plot(zono.affine_map(Z_cons_arr[k], C_flip), ax=ax, color='b', alpha=0.05, edgecolor='b', linewidth=1.0)

            # MPC solution
            x_vec = np.array(x_traj)
            ax.plot(x_vec[:,1], x_vec[:,0], '.r', markersize=3)

            # axes
            ax.set_xlabel('$x$ [m]')
            ax.set_ylabel('$y$ [m]')
            ax.axis('equal')
            ax.grid(alpha=0.2)

            # save
            if is_latex_installed():
                plt.savefig('mpc_time_varying_cons.pgf')

            plt.show()

    return x_traj, u_traj, sol
    

### run example ###
if not do_method_comparison:
    x_traj, u_traj, sol = solve_example(make_plot=True)

else:

    # sample factors
    sample_factor_arr = []
    for i in range(1,21+1):
        sample_factor_arr.append(i)

    # solver options: 'zonoopt', 'osqp', 'gurobi'
    solver_flag_arr = ['zonoopt', 'osqp', 'gurobi']

    # representation options
    use_hrep_arr = [False, True]

    # solution struct logging
    sol_log = pd.DataFrame()

    # run example in loop
    for sample_factor in sample_factor_arr:
        for solver_flag in solver_flag_arr:
            for use_hrep in use_hrep_arr:

                # solve
                try:
                    x_traj, u_traj, sol = solve_example(make_plot=False)
                except ValueError as e:
                    continue

                # log solution
                new_element = pd.DataFrame({'sample_factor': sample_factor,
                                            'solver_flag': solver_flag,
                                            'use_hrep': use_hrep,
                                            'sol': sol}, index=[0])
                sol_log = pd.concat([sol_log, new_element], ignore_index=True)

    # plot solution times
    textwidth_pt = 10.
    if is_latex_installed():
        rc_context = {
            "text.usetex": True,
            "font.size": textwidth_pt,
            "font.family": "serif",  # Choose a serif font like 'Times New Roman' or 'Computer Modern'
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
        }
    else:
        print("LaTeX not installed, using default font.")
        rc_context = {
            "font.size": textwidth_pt,
        }

    inches_per_pt = 1 / 72.27
    figsize = (245.71 * inches_per_pt, 0.9*245.71 * inches_per_pt)  # Convert pt to inches

    # plot
    with plt.rc_context(rc_context):

        fig = plt.figure(constrained_layout=True, figsize=figsize)
        ax = fig.add_subplot(111)

        # get solution time vs sample factor for each solver
        for solver_flag in solver_flag_arr:
            for use_hrep in use_hrep_arr:

                # non-existent case
                if solver_flag == 'zonoopt' and use_hrep:
                    continue

                # label
                if solver_flag == 'zonoopt':
                    solver_str = r'ZonoOpt'
                elif solver_flag == 'osqp':
                    solver_str = r'OSQP'
                elif solver_flag == 'gurobi':
                    solver_str = r'Gurobi'
                
                if use_hrep:
                    method_str = r'H-rep'
                else:
                    method_str = r'CZ'

                label = solver_str + r' ' + method_str

                # filter
                sol_log_filt = sol_log[(sol_log['solver_flag'] == solver_flag) & (sol_log['use_hrep'] == use_hrep)]

                # plot
                ax.plot(sol_log_filt['sample_factor']*55, 1000.*sol_log_filt['sol'].apply(lambda x: x.run_time), label=label)

        # labels
        ax.set_xlabel(r'Horizon $N$')
        ax.set_ylabel(r'Solution time [ms]')
        ax.legend()

        # x ticks
        xticks = [55*i for i in range(1,21+1)]
        xticklabels = []
        for i in range(1,21+1):
            if (i-1) % 5 == 0:
                xticklabels.append(str(55*i))
            else:
                xticklabels.append('')
        ax.set_xticks(xticks, xticklabels)

        # grid on
        ax.grid(alpha=0.2)

        # save
        if is_latex_installed():
            plt.savefig('solution_method_comparison.pgf')

        # display
        plt.show()

