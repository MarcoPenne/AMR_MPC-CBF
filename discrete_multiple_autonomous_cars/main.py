from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import SX, vertcat, sin, cos, Function
import numpy as np
import scipy.linalg
from CarModel import CarModel
from Path import Path
import time
import matplotlib.pyplot as plt
from utils import *

Tf = 2.5  # prediction horizon
N = int(Tf*50)  # number of discretization steps
T = 120.00  # maximum simulation time[s]
sref_N1 = Tf*0.6  # reference for final reference progress
sref_N2 = Tf*0.2
sref_N3 = Tf*0.4

l1 = 0.
l2 = 0.2
l3 = -0.2

n_lap = 10

path = Path(6, 2, 1.5)

fixed_obstacles = np.array([[2., 0.1, 0.],
                            [5., -0.1, 0.],
                            [8., 0.1, 0.],
                            [11., -0.1, 0.],
                            [14., -0.1, 0.],
                            [17., 0.1, 0.],
                            [20., -0.1, 0.]])
#fixed_obstacles = None

#moving_obstacles = np.array([5., 0.1, 0., 1., 15., -0.1, 0., 1.])

car_model = CarModel(path, 0.5*0.75, 0.25*0.75, fixed_obstacles, Tf/float(N), n_lap)
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
Q = np.diag([ 10, 1, 0, 10, 1, 0, 10, 1, 0])
R = np.eye(nu)*1e-1

Qe = np.diag([ 10, 1, 0, 10, 1, 0, 10, 1, 0])

ocp.cost.cost_type = "LINEAR_LS"
ocp.cost.cost_type_e = "LINEAR_LS"
unscale = N / Tf

ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
ocp.cost.W_e = Qe / unscale

Vx = np.zeros((ny, nx))
Vx[:nx, :nx] = np.eye(nx)
ocp.cost.Vx = Vx

Vu = np.zeros((ny, nu))
Vu[-nu:, :nu] = np.eye(nu)
ocp.cost.Vu = Vu

Vx_e = np.zeros((ny_e, nx))
Vx_e[:nx, :nx] = np.eye(nx)
ocp.cost.Vx_e = Vx_e

# set intial references
ocp.cost.yref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
ocp.cost.yref_e = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

#ocp.parameter_values = moving_obstacles

# setting constraints
ocp.constraints.lbx = np.array([-1, -1, -1])
ocp.constraints.ubx = np.array([1, 1, 1])
ocp.constraints.idxbx = np.array([1, 4, 7])
ocp.constraints.lbu = np.array([-4, -2, -4, -2, -4, -2])
ocp.constraints.ubu = np.array([4, 2, 4, 2, 4, 2])
ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])

#  Set CBF
#nh = 0
#if fixed_obstacles!=None:
print(f"Barrier functions: {model.con_h_expr.shape[0]}")
ocp.constraints.lh = np.zeros(model.con_h_expr.shape[0])
ocp.constraints.uh = np.ones(model.con_h_expr.shape[0])*1e15

# set intial condition
x0 = np.array(  [0., 0., 0.*np.pi/180, 
                12., 0.5, 0.*np.pi/180,
                6., -0.5, 0.*np.pi/180])
ocp.constraints.x0 = x0

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
#simObs_position = np.ndarray((Nsim, 1, 8))
#print(simObs_position.shape)
s01 = x0[0]
s02 = x0[3]
s03 = x0[6]
tcomp_sum = 0
tcomp_max = 0

# simulate
for i in range(Nsim):
    # update reference
    sref1 = s01 + sref_N1
    sref2 = s02 + sref_N2
    sref3 = s03 + sref_N3
    #sref_obs1 = moving_obstacles[0] + Tf*moving_obstacles[3]
    #sref_obs2 = moving_obstacles[4] + Tf*moving_obstacles[7]
    for j in range(N):
        yref = np.array([s01 + (sref1 - s01) * j / N, l1, 0, 
                        s02 + (sref2 - s02) * j / N, l2, 0,
                        s03 + (sref3 - s03) * j / N, l3, 0, 
                        0, 0, 0, 0, 0, 0])
        
        #p = np.copy(moving_obstacles)
        #p[0] += (sref_obs1 - moving_obstacles[0]) * j / N
        #p[4] += (sref_obs2 - moving_obstacles[4]) * j / N

        acados_solver.set(j, "yref", yref)
        #acados_solver.set(j, "p", p)
    yref_N = np.array([sref1, l1, 0,
                    sref2, l2, 0,
                    sref3, l3, 0])
    #p_N = np.array([sref_obs1, moving_obstacles[1], moving_obstacles[2], moving_obstacles[3], sref_obs2, moving_obstacles[5], moving_obstacles[6], moving_obstacles[7]])

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
    s01 = s01 + sref_N1/N
    s02 = s02 + sref_N2/N
    s03 = s03 + sref_N3/N
    #simObs_position[i, 0, :] = np.copy(moving_obstacles)
    #moving_obstacles[0] += (sref_obs1 - moving_obstacles[0])/ N
    #moving_obstacles[4] += (sref_obs2 - moving_obstacles[4])/ N

t = np.linspace(0.0, Nsim * Tf / N, Nsim)

time_now = datetime.datetime.now()
folder = time_now.strftime("%Y_%m_%d_%H:%M:%S")
os.mkdir('results/' + folder)

plotRes(simX, simU, t)
plt.savefig('results/' + folder + "/plots.png")
#plt.show()

# THIS IS A BIT SLOW
renderVideo(simX, simU, simX_horizon, t, car_model, fixed_obstacles, None, path, folder)