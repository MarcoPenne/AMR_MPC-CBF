from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import SX, vertcat, sin, cos, Function
import numpy as np
import scipy.linalg
from CarModel import *
from Path import Path
import time
import matplotlib.pyplot as plt
from utils import *

Tf = 2.  # prediction horizon
N = int(Tf*50)  # number of discretization steps
T = 0.5  # maximum simulation time[s]
v1 = 2.5
v2 = 2.
v3 = 1.5

sref_N1 = Tf*v1  # reference for final reference progress
sref_N2 = Tf*v2
sref_N3 = Tf*v3

n_lap = 10

path = Path(10, 5, 2)

fixed_obstacles = np.array([[6., 0.1, 0.],
                            [13., -0.1, 0.],
                            [20., 0.2, 0.],
                            [25., -0.1, 0.],
                            [30., -0.1, 0.],
                            [35., 0.1, 0.],
                            [41., -0.1, 0.]])
#fixed_obstacles = None

#moving_obstacles = np.array([5., 0.1, 0., 1., 15., -0.1, 0., 1.])
l1 = 0.
l2 = 0.5
l3 = -0.5
x01 = np.array([15., l1, 0.])
x02 = np.array([4., l2, 0.])
x03 = np.array([0., l3, 0.])

gamma = 0.5
h_cbf = 3
acados_solver1, car_model1 = create_problem(path, fixed_obstacles, N, Tf, n_lap, x01, "1", gamma, h_cbf)
acados_solver2, car_model2 = create_problem(path, fixed_obstacles, N, Tf, n_lap, x02, "2", gamma, h_cbf)
acados_solver3, car_model3 = create_problem(path, fixed_obstacles, N, Tf, n_lap, x03, "3", gamma, h_cbf)

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
    print(f'x01 = {x01}', file=f)
    print(f'x02 = {x02}', file=f)
    print(f'x03 = {x03}', file=f)
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
nx = 3
nu = 2
# initialize data structs
simX = np.ndarray((Nsim, nx*3))
simU = np.ndarray((Nsim, nu*3))
simX_horizon = np.ndarray((Nsim, N, nx*3))
#simObs_position = np.ndarray((Nsim, 1, 8))
#print(simObs_position.shape)
s01 = x01[0]
s02 = x02[0]
s03 = x03[0]
tcomp_sum = 0
tcomp_max = 0

# simulate
for i in range(Nsim):
    # update reference
    sref1 = s01 + sref_N1
    sref2 = s02 + sref_N2
    sref3 = s03 + sref_N3

    for j in range(N):
        yref1 = np.array([s01 + (sref1 - s01) * j / N, l1, 0, 0, 0])
        yref2 = np.array([s02 + (sref2 - s02) * j / N, l2, 0, 0, 0])
        yref3 = np.array([s03 + (sref3 - s03) * j / N, l3, 0, 0, 0])
        
        #s, l, theta, v
        p1 = np.array([s02 + (sref2 - s02) * j / N, x02[1], 0., v2,
                        s03 + (sref3 - s03) * j / N, x03[1], 0., v3])
        p2 = np.array([s01 + (sref1 - s01) * j / N, x01[1], 0., v1,
                        s03 + (sref3 - s03) * j / N, x03[1], 0., v3])
        p3 = np.array([s02 + (sref2 - s02) * j / N, x02[1], 0., v2,
                        s01 + (sref1 - s01) * j / N, x01[1], 0., v1])

        acados_solver1.set(j, "yref", yref1)
        acados_solver1.set(j, "p", p1)
        acados_solver2.set(j, "yref", yref2)
        acados_solver2.set(j, "p", p2)
        acados_solver3.set(j, "yref", yref3)
        acados_solver3.set(j, "p", p3)
    
    yref_N1 = np.array([sref1, l1, 0])
    yref_N2 = np.array([sref2, l2, 0])
    yref_N3 = np.array([sref3, l3, 0])
    p_N1 = np.array([sref2, x02[1], 0., v2, sref3, x03[1], 0., v3])
    p_N2 = np.array([sref1, x01[1], 0., v1, sref3, x03[1], 0., v3])
    p_N3 = np.array([sref2, x02[1], 0., v2, sref1, x01[1], 0., v1])

    acados_solver1.set(N, "yref", yref_N1)
    acados_solver1.set(N, "p", p_N1)
    acados_solver2.set(N, "yref", yref_N2)
    acados_solver2.set(N, "p", p_N2)
    acados_solver3.set(N, "yref", yref_N3)
    acados_solver3.set(N, "p", p_N3)

    # solve ocp
    t = time.time()

    status = acados_solver1.solve()
    if status != 0:
        print("acados (problem 1) returned status {} in closed loop iteration {}.".format(status, i))
    status = acados_solver2.solve()
    if status != 0:
        print("acados (problem 2) returned status {} in closed loop iteration {}.".format(status, i))
    status = acados_solver3.solve()
    if status != 0:
        print("acados (problem 3) returned status {} in closed loop iteration {}.".format(status, i))
        

    elapsed = time.time() - t

    # manage timings
    tcomp_sum += elapsed
    if elapsed > tcomp_max:
        tcomp_max = elapsed

    # get solution
    x01 = acados_solver1.get(0, "x")
    x02 = acados_solver2.get(0, "x")
    x03 = acados_solver3.get(0, "x")
    u01 = acados_solver1.get(0, "u")
    u02 = acados_solver2.get(0, "u")
    u03 = acados_solver3.get(0, "u")
    
    for j in range(nx):
        simX[i, j] = x01[j]
    for j in range(nx):
        simX[i, j+nx] = x02[j]
    for j in range(nx):
        simX[i, j+2*nx] = x03[j]

    for j in range(nu):
        simU[i, j] = u01[j]
    for j in range(nu):
        simU[i, j+nu] = u02[j]
    for j in range(nu):
        simU[i, j+2*nu] = u03[j]

    for j in range(N):
        simX_horizon[i, j, 0:3] = acados_solver1.get(j, 'x')
    for j in range(N):
        simX_horizon[i, j, 3:6] = acados_solver2.get(j, 'x')
    for j in range(N):
        simX_horizon[i, j, 6:9] = acados_solver3.get(j, 'x')

    # update initial condition
    x01 = acados_solver1.get(1, "x")
    acados_solver1.set(0, "lbx", x01)
    acados_solver1.set(0, "ubx", x01)
    s01 = x01[0]

    x02 = acados_solver2.get(1, "x")
    acados_solver2.set(0, "lbx", x02)
    acados_solver2.set(0, "ubx", x02)
    s02 = x02[0]

    x03 = acados_solver3.get(1, "x")
    acados_solver3.set(0, "lbx", x03)
    acados_solver3.set(0, "ubx", x03)
    s03 = x03[0]

    #simObs_position[i, 0, :] = np.copy(moving_obstacles)
    #moving_obstacles[0] += (sref_obs1 - moving_obstacles[0])/ N
    #moving_obstacles[4] += (sref_obs2 - moving_obstacles[4])/ N

with open('results/'+folder+'/data.txt', 'a') as f:
    print(f'computation_time = {tcomp_sum}', file=f)


t = np.linspace(0.0, Nsim * Tf / N, Nsim)


plotRes(simX, simU, t)
plt.savefig('results/' + folder + "/plots.png")
#plt.show()

# THIS IS A BIT SLOW
renderVideo(simX, simU, simX_horizon, t, car_model1, fixed_obstacles, None, path, folder, h_cbf)