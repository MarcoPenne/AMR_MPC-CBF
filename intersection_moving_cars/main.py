from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import SX, vertcat, sin, cos, Function
import numpy as np
import scipy.linalg
from CarModel import *
from Path import Path
import time
import matplotlib.pyplot as plt
from new_utils import *

Tf = 2.  # prediction horizon
N = int(Tf*50)  # number of discretization steps
T = 0.50  # maximum simulation time[s]
v1 = 2.
v2 = 1.5
sref_N1 = Tf*v1  # reference for final reference progress
sref_N2 = Tf*v2  # reference for final reference progress

n_lap = 10

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
fixed_obstacles1 = None#np.array([[7., 0.4, 0.],[13., -0.5, 0.]])
fixed_obstacles2 = None#np.array([[15., 0.4, 0.],[27.5, -0.4, 0.]])
 
moving_obstacles1 = np.array([path1.get_len()/2 - np.pi - l2/2, 0.3, 0., 0.5, 
                            path1.get_len() - np.pi - l2/2, 0.3, 0., 0.5])
moving_obstacles2 = np.array([l2/2 + path2.get_len()/2, 0.3, 0., 0.5,
                            l2/2 + path2.get_len(), -0.3, 0.0, 0.5])

x01 = np.array([0., 0., 0.])
x02 = np.array([15.,  0.,  0.])

gamma = 0.5
h_cbf = 1.
acados_solver1, car_model1 = create_problem(path1, 1, 0.5, fixed_obstacles1, path2, fixed_obstacles2, N, Tf, n_lap, x01, "1", gamma, h_cbf)
acados_solver2, car_model1 = create_problem(path2, 1, 0.5, fixed_obstacles2, path1, fixed_obstacles1, N, Tf, n_lap, x02, "2", gamma, h_cbf)

# Create log file
time_now = datetime.datetime.now()
folder = time_now.strftime("%Y_%m_%d_%H:%M:%S")
os.mkdir('results/' + folder)
with open('results/'+folder+'/data.txt', 'w') as f:
    print(f"# {os.getcwd().split('/')[-1]}", file=f)
    print(f'Tf = {Tf}', file=f)
    print(f'v1 = {v1}', file=f)
    print(f'v2 = {v2}', file=f)
    print(f'moving_obstacles1 = {moving_obstacles1}', file=f)
    print(f'moving_obstacles2 = {moving_obstacles2}', file=f)
    print(f'x01 = {x01}', file=f)
    print(f'x02 = {x02}', file=f)
    print(f'gamma = {gamma}', file=f)
    print(f'h_cbf = {h_cbf}', file=f)
    print(f'Q = {(acados_solver1.acados_ocp.cost.W * (Tf/N))[:3, :3]}', file=f)
    print(f'R = {(acados_solver1.acados_ocp.cost.W * (Tf/N))[3:, 3:]}', file=f)
    print(f'Q_e = {acados_solver1.acados_ocp.cost.W_e * (N/Tf)}', file=f)
    print(f'qp_solver = {acados_solver1.acados_ocp.solver_options.qp_solver}', file=f)
    print(f'nlp_solver_type = {acados_solver1.acados_ocp.solver_options.nlp_solver_type}', file=f)
    print(f'qp_solver_iter_max = {acados_solver1.acados_ocp.solver_options.qp_solver_iter_max}', file=f)
    print(f'nlp_solver_max_iter = {acados_solver1.acados_ocp.solver_options.nlp_solver_max_iter}', file=f)

Nsim = int(T * N / Tf)
# initialize data structs
nx = 3
nu = 2
simX = np.ndarray((Nsim, nx*2))
simU = np.ndarray((Nsim, nu*2))
simX_horizon = np.ndarray((Nsim, N, nx*2))
simObs_position1 = np.ndarray((Nsim, 1, 8))
simObs_position2 = np.ndarray((Nsim, 1, 8))
s01 = x01[0]
s02 = x02[0]
tcomp_sum = 0
tcomp_max = 0
time_iterations = np.zeros(Nsim)
cost_integral1 = 0
cost_integral2 = 0

# simulate
for i in range(Nsim):
    # update reference
    sref1 = s01 + sref_N1
    sref2 = s02 + sref_N2
    sref_obs1 = moving_obstacles1[0] + Tf*moving_obstacles1[3]
    sref_obs2 = moving_obstacles1[4] + Tf*moving_obstacles1[7]
    sref_obs3 = moving_obstacles2[0] + Tf*moving_obstacles2[3]
    sref_obs4 = moving_obstacles2[4] + Tf*moving_obstacles2[7]
    for j in range(N):
        yref = np.array([s01 + (sref1 - s01) * j / N, 0, 0, v1, 0])
        acados_solver1.set(j, "yref", yref)
        yref = np.array([s02 + (sref2 - s02) * j / N, 0, 0, v2, 0])
        acados_solver2.set(j, "yref", yref)

        p = np.zeros(20)
        p[0:8] = np.copy(moving_obstacles1) 
        p[8:16] = np.copy(moving_obstacles2)
        p[16:19] = np.copy(x02)
        p[19] = v2
        p[0] += (sref_obs1 - moving_obstacles1[0]) * j / N
        p[4] += (sref_obs2 - moving_obstacles1[4]) * j / N
        p[8] += (sref_obs3 - moving_obstacles2[0]) * j / N
        p[12] += (sref_obs4 - moving_obstacles2[4]) * j / N
        p[16] += (sref2 - s02) * j / N
        
        P = np.zeros_like(p)
        P[0], P[1] = path1.get_cartesian_coords(p[0], p[1])
        P[2] = path1.get_theta_r(p[0])
        P[3] = p[3]
        P[4], P[5] = path1.get_cartesian_coords(p[4], p[5])
        P[6] = path1.get_theta_r(p[4])
        P[7] = p[7]
        P[8], P[9] = path2.get_cartesian_coords(p[8], p[9])
        P[10] = path2.get_theta_r(p[8])
        P[11] = p[11]
        P[12], P[13] = path2.get_cartesian_coords(p[12], p[13])
        P[14] = path2.get_theta_r(p[12])
        P[15] = p[15]
        P[16], P[17] = path2.get_cartesian_coords(p[16], p[17])
        P[18] = path2.get_theta_r(p[16])
        P[19] = p[19]
        acados_solver1.set(j, "p", P)

        p = np.zeros(20)
        p[0:8] = np.copy(moving_obstacles1) 
        p[8:16] = np.copy(moving_obstacles2)
        p[16:19] = np.copy(x01)
        p[19] = v1
        p[0] += (sref_obs1 - moving_obstacles1[0]) * j / N
        p[4] += (sref_obs2 - moving_obstacles1[4]) * j / N
        p[8] += (sref_obs3 - moving_obstacles2[0]) * j / N
        p[12] += (sref_obs4 - moving_obstacles2[4]) * j / N
        p[16] += (sref1 - s01) * j / N
        
        P = np.zeros_like(p)
        P[0], P[1] = path1.get_cartesian_coords(p[0], p[1])
        P[2] = path1.get_theta_r(p[0])
        P[3] = p[3]
        P[4], P[5] = path1.get_cartesian_coords(p[4], p[5])
        P[6] = path1.get_theta_r(p[4])
        P[7] = p[7]
        P[8], P[9] = path2.get_cartesian_coords(p[8], p[9])
        P[10] = path2.get_theta_r(p[8])
        P[11] = p[11]
        P[12], P[13] = path2.get_cartesian_coords(p[12], p[13])
        P[14] = path2.get_theta_r(p[12])
        P[15] = p[15]
        P[16], P[17] = path1.get_cartesian_coords(p[16], p[17])
        P[18] = path1.get_theta_r(p[16])
        P[19] = p[19]
        acados_solver2.set(j, "p", P)

    yref_N = np.array([sref1, 0, 0])
    acados_solver1.set(N, "yref", yref_N)
    yref_N = np.array([sref2, 0, 0])
    acados_solver2.set(N, "yref", yref_N)
    
    p = np.zeros(20)
    p[0:8] = np.copy(moving_obstacles1) 
    p[8:16] = np.copy(moving_obstacles2)
    p[16:19] = np.copy(x02)
    p[19] = v2
    p[0] = sref_obs1
    p[4] = sref_obs2
    p[8] = sref_obs3
    p[12] = sref_obs4
    p[16] = sref2

    P = np.zeros_like(p)
    P[0], P[1] = path1.get_cartesian_coords(p[0], p[1])
    P[2] = path1.get_theta_r(p[0])
    P[3] = p[3]
    P[4], P[5] = path1.get_cartesian_coords(p[4], p[5])
    P[6] = path1.get_theta_r(p[4])
    P[7] = p[7]
    P[8], P[9] = path2.get_cartesian_coords(p[8], p[9])
    P[10] = path2.get_theta_r(p[8])
    P[11] = p[11]
    P[12], P[13] = path2.get_cartesian_coords(p[12], p[13])
    P[14] = path2.get_theta_r(p[12])
    P[15] = p[15]
    P[16], P[17] = path2.get_cartesian_coords(p[16], p[17])
    P[18] = path2.get_theta_r(p[16])
    P[19] = p[19]
    acados_solver1.set(N, "p", P)

    p = np.zeros(20)
    p[0:8] = np.copy(moving_obstacles1) 
    p[8:16] = np.copy(moving_obstacles2)
    p[16:19] = np.copy(x01)
    p[19] = v1
    p[0] = sref_obs1
    p[4] = sref_obs2
    p[8] = sref_obs3
    p[12] = sref_obs4
    p[16] = sref1
    
    P = np.zeros_like(p)
    P[0], P[1] = path1.get_cartesian_coords(p[0], p[1])
    P[2] = path1.get_theta_r(p[0])
    P[3] = p[3]
    P[4], P[5] = path1.get_cartesian_coords(p[4], p[5])
    P[6] = path1.get_theta_r(p[4])
    P[7] = p[7]
    P[8], P[9] = path2.get_cartesian_coords(p[8], p[9])
    P[10] = path2.get_theta_r(p[8])
    P[11] = p[11]
    P[12], P[13] = path2.get_cartesian_coords(p[12], p[13])
    P[14] = path2.get_theta_r(p[12])
    P[15] = p[15]
    P[16], P[17] = path1.get_cartesian_coords(p[16], p[17])
    P[18] = path1.get_theta_r(p[16])
    P[19] = p[19]
    acados_solver1.set(N, "p", P)

    # solve ocp
    t = time.time()

    status = acados_solver1.solve()
    if status != 0:
        print("acados (problem 1) returned status {} in closed loop iteration {}.".format(status, i))
    status = acados_solver2.solve()
    if status != 0:
        print("acados (problem 2) returned status {} in closed loop iteration {}.".format(status, i))

    elapsed = time.time() - t
    time_iterations[i] = elapsed

    # manage timings
    tcomp_sum += elapsed
    if elapsed > tcomp_max:
        tcomp_max = elapsed

    # get solution
    x01 = acados_solver1.get(0, "x")
    x02 = acados_solver2.get(0, "x")
    u01 = acados_solver1.get(0, "u")
    u02 = acados_solver2.get(0, "u")
    cost_integral1 += np.matmul(u01, u01.T)*Tf / N
    cost_integral2 += np.matmul(u02, u02.T)*Tf / N
    for j in range(nx):
        simX[i, j] = x01[j]
    for j in range(nx):
        simX[i, j+nx] = x02[j]
    for j in range(nu):
        simU[i, j] = u01[j]
    for j in range(nu):
        simU[i, j+nu] = u02[j]
    for j in range(N):
        simX_horizon[i, j, 0:3] = acados_solver1.get(j, 'x')
    for j in range(N):
        simX_horizon[i, j, 3:6] = acados_solver2.get(j, 'x')
    
    # update initial condition
    x01 = acados_solver1.get(1, "x")
    acados_solver1.set(0, "lbx", x01)
    acados_solver1.set(0, "ubx", x01)
    s01 = x01[0]

    x02 = acados_solver2.get(1, "x")
    acados_solver2.set(0, "lbx", x02)
    acados_solver2.set(0, "ubx", x02)
    s02 = x02[0]
    
    simObs_position1[i, 0, :] = np.copy(moving_obstacles1)
    simObs_position2[i, 0, :] = np.copy(moving_obstacles2)
    moving_obstacles1[0] += (sref_obs1 - moving_obstacles1[0])/ N
    moving_obstacles1[4] += (sref_obs2 - moving_obstacles1[4])/ N
    moving_obstacles2[0] += (sref_obs3 - moving_obstacles2[0])/ N
    moving_obstacles2[4] += (sref_obs4 - moving_obstacles2[4])/ N
    

with open('results/'+folder+'/data.txt', 'a') as f:
    print(f'min_time = {np.min(time_iterations)}', file=f)
    print(f'max_time = {np.max(time_iterations)}', file=f)
    print(f'mean_time = {np.mean(time_iterations)}', file=f)
    print(f'std_time = {np.std(time_iterations)}', file=f)
    print(f'cost integral1 = {cost_integral1}', file=f)
    print(f'cost integral2 = {cost_integral2}', file=f)

t = np.linspace(0.0, Nsim * Tf / N, Nsim)

plotRes(simX, simU, t)
plt.savefig('results/' + folder + "/plots.png")
plt.savefig('results/' + folder + "/plots.eps")
#plt.show()

simX1 = simX[:, :3]
simX2 = simX[:, 3:]
simU1 = simU[:, :2]
simU2 = simU[:, 2:]
simX_horizon1 = simX_horizon[:, :, :3]
simX_horizon2 = simX_horizon[:, :, 3:]
# THIS IS A BIT SLOW
renderVideo(simX1, simU1, simX_horizon1, fixed_obstacles1, simObs_position1, path1,
          simX2, simX2, simX_horizon2, fixed_obstacles2, simObs_position2, path2,
          car_model1, t, folder, h_cbf)