from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import SX, vertcat, sin, cos, Function
import numpy as np
import scipy.linalg
from CarModel import CarModel
from Path import Path
import time
import matplotlib.pyplot as plt
from utils import *
import inspect

Tf = 1.  # prediction horizon
N = int(Tf*50)  # number of discretization steps
T = 20.  # maximum simulation time[s]
v = 2.5
sref_N = Tf*v  # reference for final reference progress

n_lap = 3

path = Path(10, 5, 2)

fixed_obstacles = np.array([[6., 0.1, 0.],
                            [13., -0.1, 0.],
                            [20., 0.2, 0.],
                            [25., -0.1, 0.],
                            [30., -0.1, 0.],
                            [35., 0.1, 0.],
                            [41., -0.1, 0.]])
fixed_obstacles = None

moving_obstacles = np.array([5., 0.3, 0., 1., 15., -0.3, 0., 1.])

gamma = lambda hx: 1.5*hx
h_cbf = 3
car_model = CarModel(path, 1, 0.5, fixed_obstacles, n_lap, gamma, h_cbf)
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
Q = np.diag([ 100, 10, 10])
R = np.eye(nu)*10

Qe = np.diag([ 100, 10, 10])

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

ocp.parameter_values = moving_obstacles

# setting constraints
ocp.constraints.lbx = np.array([-2])
ocp.constraints.ubx = np.array([2])
ocp.constraints.idxbx = np.array([1])
ocp.constraints.lbu = np.array([-4, -2])
ocp.constraints.ubu = np.array([4, 2])
ocp.constraints.idxbu = np.array([0, 1])

#  Set CBF
print(f"Created {model.con_h_expr.shape[0]} CBFs")
ocp.constraints.lh = np.zeros(model.con_h_expr.shape[0])
ocp.constraints.uh = np.ones(model.con_h_expr.shape[0])*1e15

# set intial condition
x0=np.array([0., 0., 0.])
ocp.constraints.x0 = x0#[0., -1., 20.*np.pi/180.])

# set QP solver and integration
ocp.solver_options.tf = Tf
ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
#ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
ocp.solver_options.nlp_solver_type = "SQP_RTI"
ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
ocp.solver_options.integrator_type = "ERK"
ocp.solver_options.qp_solver_iter_max = 1000
ocp.solver_options.nlp_solver_max_iter = 1000
#ocp.solver_options.sim_method_num_stages = N
#ocp.solver_options.sim_method_num_steps = N

# create solver
acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

# Create log file
time_now = datetime.datetime.now()
folder = time_now.strftime("%Y_%m_%d_%H:%M:%S")
os.mkdir('results/' + folder)
with open('results/'+folder+'/data.txt', 'w') as f:
    print(f"# {os.getcwd().split('/')[-1]}", file=f)
    print(f'Tf = {Tf}', file=f)
    print(f'v = {v}', file=f)
    print(f'moving_obstacles = {moving_obstacles}', file=f)
    print(f'x0 = {x0}', file=f)
    print(f'gamma = {str(inspect.getsourcelines(gamma)[0][0].strip())}', file=f)
    print(f'h_cbf = {h_cbf}', file=f)
    print(f'Q = {Q}', file=f)
    print(f'R = {R}', file=f)
    print(f'Qe = {Qe}', file=f)
    print(f'qp_solver = {ocp.solver_options.qp_solver}', file=f)
    print(f'nlp_solver_type = {ocp.solver_options.nlp_solver_type}', file=f)
    print(f'qp_solver_iter_max = {ocp.solver_options.qp_solver_iter_max}', file=f)
    print(f'nlp_solver_max_iter = {ocp.solver_options.nlp_solver_max_iter}', file=f)

Nsim = int(T * N / Tf)
# initialize data structs
simX = np.ndarray((Nsim, nx))
simU = np.ndarray((Nsim, nu))
simX_horizon = np.ndarray((Nsim, N, nx))
simObs_position = np.ndarray((Nsim, 1, 8))
print(simObs_position.shape)
s0 = 0
tcomp_sum = 0
tcomp_max = 0
time_iterations = np.zeros(Nsim)
cost_integral = 0

# simulate
for i in range(Nsim):
    # update reference
    sref = s0 + sref_N
    sref_obs1 = moving_obstacles[0] + Tf*moving_obstacles[3]
    sref_obs2 = moving_obstacles[4] + Tf*moving_obstacles[7]
    for j in range(N):
        yref = np.array([s0 + (sref - s0) * j / N, 0, 0, v, 0])
        
        p = np.copy(moving_obstacles)
        p[0] += (sref_obs1 - moving_obstacles[0]) * j / N
        p[4] += (sref_obs2 - moving_obstacles[4]) * j / N

        acados_solver.set(j, "yref", yref)
        acados_solver.set(j, "p", p)
    yref_N = np.array([sref, 0, 0])
    p_N = np.array([sref_obs1, moving_obstacles[1], moving_obstacles[2], moving_obstacles[3], sref_obs2, moving_obstacles[5], moving_obstacles[6], moving_obstacles[7]])

    acados_solver.set(N, "yref", yref_N)
    acados_solver.set(N, "p", p_N)

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
    cost_integral += np.matmul(u0, u0.T)*Tf / N
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
    simObs_position[i, 0, :] = np.copy(moving_obstacles)
    moving_obstacles[0] += (sref_obs1 - moving_obstacles[0])/ N
    moving_obstacles[4] += (sref_obs2 - moving_obstacles[4])/ N

with open('results/'+folder+'/data.txt', 'a') as f:
    print(f'min_time = {np.min(time_iterations)}', file=f)
    print(f'max_time = {np.max(time_iterations)}', file=f)
    print(f'mean_time = {np.mean(time_iterations)}', file=f)
    print(f'std_time = {np.std(time_iterations)}', file=f)
    print(f'cost integral = {cost_integral}', file=f)

t = np.linspace(0.0, Nsim * Tf / N, Nsim)

plotRes2(simX, simU, t)
plt.savefig('results/' + folder + "/plots.eps")
plt.savefig('results/' + folder + "/plots.png", dpi=300)

plotRes3(simX, simU, t)
plt.savefig('results/' + folder + "/plots3.eps")
plt.savefig('results/' + folder + "/plots3.png", dpi=300)

plotResS(simX, simU, t)
plt.savefig('results/' + folder + "/plotsS.eps")
plt.savefig('results/' + folder + "/plotsS.png", dpi=300)
#plt.show()

with open('results/' + folder + "/simX.npy", 'wb') as f:
    np.save(f, simX)
with open('results/' + folder + "/simU.npy", 'wb') as f:
    np.save(f, simU)
with open('results/' + folder + "/simX_horizon.npy", 'wb') as f:
    np.save(f, simX_horizon)
with open('results/' + folder + "/t.npy", 'wb') as f:
    np.save(f, t)
with open('results/' + folder + "/fixed_obstacles.npy", 'wb') as f:
    np.save(f, fixed_obstacles)
with open('results/' + folder + "/simObs_position.npy", 'wb') as f:
    np.save(f, simObs_position)
#plt.show()

# THIS IS A BIT SLOW
#renderVideo(simX, simU, simX_horizon, t, car_model, fixed_obstacles, simObs_position, path, folder, h_cbf)
