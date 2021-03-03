from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import SX, vertcat, sin, cos, Function
import numpy as np
import scipy.linalg
from CarModel import CarModel
from Path import Path
import time
import matplotlib.pyplot as plt
from utils import *

Tf = 2.  # prediction horizon
N = int(Tf*50)  # number of discretization steps
T = 0.5  # maximum simulation time[s]
v1 = 3.
v2 = 2.
v3 = 1.
sref_N1 = Tf*v1  # reference for final reference progress
sref_N2 = Tf*v2
sref_N3 = Tf*v3


n_lap = 10

path = Path(10, 5, 2)

fixed_obstacles = np.array([[6., 0.1, 0.],
                            [20., 0.2, 0.],
                            [35., 0.1, 0.]])

# fixed_obstacles = np.array([[6., 0.1, 0.],
#                             [13., -0.1, 0.],
#                             [20., 0.2, 0.],
#                             [25., -0.1, 0.],
#                             [30., -0.1, 0.],
#                             [35., 0.1, 0.],
#                             [41., -0.1, 0.]])
#fixed_obstacles = None

#moving_obstacles = np.array([5., 0.1, 0., 1., 15., -0.1, 0., 1.])
gamma = 0.5
h_cbf = 3.
car_model = CarModel(path, 1., 0.5, fixed_obstacles, Tf/float(N), n_lap, gamma, h_cbf)
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
Q = np.diag([ 100, 10, 10, 100, 10, 10, 100, 10, 10])
R = np.eye(nu)*10

Qe = np.diag([ 100, 10, 10, 100, 10, 10, 100, 10, 10])

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
l1 = 0.
l2 = 0.5
l3 = -0.5
x0 = np.array(  [15., l1, 0., 
                4., l2, 0.,
                0., l3, 0.])
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
#ocp.solver_options.qp_solver_iter_max = 800
#ocp.solver_options.nlp_solver_max_iter = 800

# create solver
acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

# Create log file
time_now = datetime.datetime.now()
folder = time_now.strftime("%Y_%m_%d_%H:%M:%S")
os.mkdir('results/' + folder)
with open('results/'+folder+'/data.txt', 'w') as f:
    print(f"# {os.getcwd().split('/')[-1]}", file=f)
    print(f'Tf = {Tf}', file=f)
    print(f'v1 = {v1}', file=f)
    print(f'v2 = {v2}', file=f)
    print(f'v3 = {v3}', file=f)
    print(f'fixed_obstacles = {fixed_obstacles}', file=f)
    print(f'x0 = {x0}', file=f)
    print(f'gamma = {gamma}', file=f)
    print(f'h_cbf = {h_cbf}', file=f)
    print(f'Q = {(acados_solver.acados_ocp.cost.W * (Tf/N))[:9, :9]}', file=f)
    print(f'R = {(acados_solver.acados_ocp.cost.W * (Tf/N))[9:, 9:]}', file=f)
    print(f'Q_e = {acados_solver.acados_ocp.cost.W_e * (N/Tf)}', file=f)
    print(f'qp_solver = {acados_solver.acados_ocp.solver_options.qp_solver}', file=f)
    print(f'nlp_solver_type = {acados_solver.acados_ocp.solver_options.nlp_solver_type}', file=f)
    print(f'qp_solver_iter_max = {acados_solver.acados_ocp.solver_options.qp_solver_iter_max}', file=f)
    print(f'nlp_solver_max_iter = {acados_solver.acados_ocp.solver_options.nlp_solver_max_iter}', file=f)

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
time_iterations = np.zeros(Nsim)
cost_integral1 = 0
cost_integral2 = 0
cost_integral3 = 0

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
                        v1, 0, v2, 0, v3, 0])
        
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
    time_iterations[i] = elapsed

    # manage timings
    tcomp_sum += elapsed
    if elapsed > tcomp_max:
        tcomp_max = elapsed

    # get solution
    x0 = acados_solver.get(0, "x")
    u0 = acados_solver.get(0, "u")
    cost_integral1 += np.matmul(u0[0:2], u0[0:2].T)*Tf / N
    cost_integral2 += np.matmul(u0[2:4], u0[2:4].T)*Tf / N
    cost_integral3 += np.matmul(u0[4:6], u0[4:6].T)*Tf / N
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

with open('results/'+folder+'/data.txt', 'a') as f:
    print(f'min_time = {np.min(time_iterations)}', file=f)
    print(f'max_time = {np.max(time_iterations)}', file=f)
    print(f'mean_time = {np.mean(time_iterations)}', file=f)
    print(f'std_time = {np.std(time_iterations)}', file=f)
    print(f'cost integral1 = {cost_integral1}', file=f)
    print(f'cost integral2 = {cost_integral2}', file=f)
    print(f'cost integral3 = {cost_integral3}', file=f)

t = np.linspace(0.0, Nsim * Tf / N, Nsim)

plotRes(simX, simU, t)
plt.savefig('results/' + folder + "/plots.eps")
plt.savefig('results/' + folder + "/plots.png")
#plt.show()

# THIS IS A BIT SLOW
renderVideo(simX, simU, simX_horizon, t, car_model, fixed_obstacles, None, path, folder, h_cbf)