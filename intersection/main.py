from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import SX, vertcat, sin, cos, Function
import numpy as np
import scipy.linalg
from CarModel import CarModel
from Path import Path
import time
import matplotlib.pyplot as plt
from new_utils import *

Tf = 1.5  # prediction horizon
N = int(Tf*50)  # number of discretization steps
T = 30.00  # maximum simulation time[s]
sref_N = Tf*2.5  # reference for final reference progress

n_lap = 3

l1=10
l2 = 3
path1 = Path(l1, l2, 2)
path2 = Path(l2, l1, 2, traslx=(l1-l2)/2, trasly=-(l1-l2)/2)

fixed_obstacles1 = np.array([[6., 0.1, 0.],
                            [13., -0.1, 0.],
                            [20., 0.2, 0.],
                            [25., -0.1, 0.],
                            [30., -0.1, 0.],
                            [35., 0.1, 0.],
                            [41., -0.1, 0.]])
fixed_obstacles1 = np.array([[7., 0.4, 0.],[13., -0.5, 0.]])
fixed_obstacles2 = np.array([[15., 0.4, 0.],[27.5, -0.4, 0.]])
 
moving_obstacles1 = None#np.array([5., 0.1, 0., 1., 15., -0.1, 0., 1.])
moving_obstacles2 = None#np.array([5., 0.0, 0.0, 1.])

car_model = CarModel(path1, 1, 0.5, fixed_obstacles1, path2, fixed_obstacles2, Tf/float(N), n_lap)
model = car_model.model
ocp = AcadosOcp()
ocp.model = model
print(model)

# set dimensions
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx

ocp.dims.N = N

# set cost
Q = np.diag([ 10, 1, 0])
R = np.eye(nu)*1e-1

Qe = np.diag([ 10, 1, 1])

ocp.cost.cost_type = "LINEAR_LS"
ocp.cost.cost_type_e = "LINEAR_LS"
unscale = N / Tf

ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
ocp.cost.W_e = Qe / unscale

Vx = np.zeros((ny, nx))
Vx[:nx, :nx] = np.eye(nx)
ocp.cost.Vx = Vx

Vu = np.zeros((ny, nu))
Vu[3, 0] = 1.0
Vu[4, 1] = 1.0
ocp.cost.Vu = Vu

Vx_e = np.zeros((ny_e, nx))
Vx_e[:nx, :nx] = np.eye(nx)
ocp.cost.Vx_e = Vx_e

# set intial references
ocp.cost.yref = np.array([0, 0, 0, 0, 0])
ocp.cost.yref_e = np.array([0, 0, 0])

#x_obs, y_obs = path2.get_cartesian_coords(moving_obstacles2[0], moving_obstacles2[1])
#theta_obs = path2.get_theta_r(moving_obstacles2[0])
#P = np.array([x_obs, y_obs, theta_obs, moving_obstacles2[3]])
#ocp.parameter_values = P

# setting constraints
ocp.constraints.lbx = np.array([-2])
ocp.constraints.ubx = np.array([2])
ocp.constraints.idxbx = np.array([1])
ocp.constraints.lbu = np.array([-4, -2])
ocp.constraints.ubu = np.array([4, 2])
ocp.constraints.idxbu = np.array([0, 1])

#  Set CBF
if model.con_h_expr is not None:
    ocp.constraints.lh = np.zeros(model.con_h_expr.shape[0])
    ocp.constraints.uh = np.ones(model.con_h_expr.shape[0])*1e15

# set intial condition
ocp.constraints.x0 = np.array([0., 0., 0.])

# set QP solver and integration
ocp.solver_options.tf = Tf
# ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
ocp.solver_options.nlp_solver_type = "SQP_RTI"
ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
ocp.solver_options.integrator_type = "DISCRETE"
ocp.solver_options.sim_method_num_stages = 4
ocp.solver_options.sim_method_num_steps = 3

# create solver
acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

Nsim = int(T * N / Tf)
# initialize data structs
simX = np.ndarray((Nsim, nx))
simU = np.ndarray((Nsim, nu))
simX_horizon = np.ndarray((Nsim, N, nx))
simObs_position2 = np.ndarray((Nsim, 1, 4))
simObs_position2 = None
s0 = 0
tcomp_sum = 0
tcomp_max = 0

# simulate
for i in range(Nsim):
    # update reference
    sref = s0 + sref_N
    #sref_obs1 = moving_obstacles2[0] + Tf*moving_obstacles2[3]
    #sref_obs2 = moving_obstacles[4] + Tf*moving_obstacles[7]
    for j in range(N):
        yref = np.array([s0 + (sref - s0) * j / N, 0, 0, 0, 0])
        
        #p = np.copy(moving_obstacles2)
        #p[0] += (sref_obs1 - moving_obstacles2[0]) * j / N
        #p[4] += (sref_obs2 - moving_obstacles[4]) * j / N

        acados_solver.set(j, "yref", yref)
        #x_obs, y_obs = path2.get_cartesian_coords(p[0], p[1])
        #theta_obs = path2.get_theta_r(p[0])
        #P = np.array([x_obs, y_obs, theta_obs, p[3]])
        #acados_solver.set(j, "p", p)

    yref_N = np.array([sref, 0, 0])
    
    #x_obs, y_obs = path2.get_cartesian_coords(sref_obs1, moving_obstacles2[1])
    #theta_obs = path2.get_theta_r(sref_obs1)
    #p_N = np.array([x_obs, y_obs, theta_obs, moving_obstacles2[3]])

    acados_solver.set(N, "yref", yref_N)
    #acados_solver.set(N, "p", p_N)

    # solve ocp
    t = time.time()

    status = acados_solver.solve()
    if status != 0:
        print("acados returned status {} in closed loop iteration {}.".format(status, i))

    elapsed = time.time() - t

    # manage timings
    tcomp_sum += elapsed
    if elapsed > tcomp_max:
        tcomp_max = elapsed

    # get solution
    x0 = acados_solver.get(0, "x")
    u0 = acados_solver.get(0, "u")
    for j in range(nx):
        simX[i, j] = x0[j]
    for j in range(nu):
        simU[i, j] = u0[j]
    for j in range(N):
        simX_horizon[i, j, :] = acados_solver.get(j, 'x')

    # update initial condition
    x0 = acados_solver.get(1, "x")
    acados_solver.set(0, "lbx", x0)
    acados_solver.set(0, "ubx", x0)
    s0 = x0[0]
    #simObs_position2[i, 0, :] = np.copy(moving_obstacles2)
    #moving_obstacles2[0] += (sref_obs1 - moving_obstacles2[0])/ N
    #moving_obstacles[4] += (sref_obs2 - moving_obstacles[4])/ N

t = np.linspace(0.0, Nsim * Tf / N, Nsim)

time_now = datetime.datetime.now()
folder = time_now.strftime("%Y_%m_%d_%H:%M:%S")
os.mkdir('results/' + folder)

plotRes(simX, simU, t)
plt.savefig('results/' + folder + "/plots.png")
#plt.show()

# THIS IS A BIT SLOW
renderVideo(simX, simU, simX_horizon, fixed_obstacles1, None, path1,
            None, None, None, fixed_obstacles2, simObs_position2, path2,
            car_model, t, folder)