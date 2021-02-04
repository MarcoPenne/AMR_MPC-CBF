from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import SX, vertcat, sin, cos, Function
import numpy as np
import scipy.linalg
from CarModel import CarModel
from Path import Path
import time
import matplotlib.pyplot as plt
from utils import *

Tf = 1.5  # prediction horizon
N = int(Tf*50)  # number of discretization steps
T = 30.00  # maximum simulation time[s]
sref_N = Tf*2  # reference for final reference progress

path = Path(10, 5, 2)
car_model = CarModel(path, 1, 0.5, 0.02)
model = car_model.model
ocp = AcadosOcp()
ocp.model = model
print(model)

# define constraint
#model_ac.con_h_expr = constraint.expr

# set dimensions
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx

ocp.dims.N = N
#ns = 2
#nsh = 2

# set cost
Q = np.diag([ 10, 1, 0])
R = np.eye(nu)*1e-1
#R[0, 0] = 1
#R[1, 1] = 1

Qe = np.diag([ 10, 1, 0])

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


#

#ocp.cost.zl = 100 * np.ones((ns,))
#ocp.cost.Zl = 0 * np.ones((ns,))
#ocp.cost.zu = 100 * np.ones((ns,))
#ocp.cost.Zu = 0 * np.ones((ns,))

# set intial references
ocp.cost.yref = np.array([0, 0, 0, 0, 0])
ocp.cost.yref_e = np.array([0, 0, 0])

# setting constraints
ocp.constraints.lbx = np.array([-2])
ocp.constraints.ubx = np.array([2])
ocp.constraints.idxbx = np.array([1])
ocp.constraints.lbu = np.array([-4, -2])
ocp.constraints.ubu = np.array([4, 2])
ocp.constraints.idxbu = np.array([0, 1])
# ocp.constraints.lsbx=np.zero s([1])
# ocp.constraints.usbx=np.zeros([1])
# ocp.constraints.idxsbx=np.array([1])

#  Set CBF
#ocp.model.con_h_expr = 

ocp.constraints.lh = np.array(
    [
        0
    ]
)
ocp.constraints.uh = np.array(
    [
        1e15
    ]
)
# ocp.constraints.lsh = np.zeros(nsh)
# ocp.constraints.ush = np.zeros(nsh)
# ocp.constraints.idxsh = np.array([0, 2])

# set intial condition
ocp.constraints.x0 = np.array([0, 0, 0])

# set QP solver and integration
ocp.solver_options.tf = Tf
# ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
ocp.solver_options.nlp_solver_type = "SQP_RTI"
ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
ocp.solver_options.integrator_type = "ERK"
ocp.solver_options.sim_method_num_stages = 4
ocp.solver_options.sim_method_num_steps = 3

# ocp.solver_options.qp_solver_tol_stat = 1e-2
# ocp.solver_options.qp_solver_tol_eq = 1e-2
# ocp.solver_options.qp_solver_tol_ineq = 1e-2
# ocp.solver_options.qp_solver_tol_comp = 1e-2

# create solver
acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

# Tf = 1.0
# nx = model.x.size()[0]
# print(nx)
# nu = model.u.size()[0]
# print(nu)
# ny = nx + nu
# ny_e = nx
# N = 10

# # set dimensions
# ocp.dims.N = N

# # set cost module
# ocp.cost.cost_type = 'LINEAR_LS'
# ocp.cost.cost_type_e = 'LINEAR_LS'

# Q = 2*np.diag([1e3, 1e3, 1e-2])
# R = 2*np.diag([1e-2, 1e-2])

# ocp.cost.W = scipy.linalg.block_diag(Q, R)

# ocp.cost.W_e = Q

# ocp.cost.Vx = np.zeros((ny, nx))
# ocp.cost.Vx[:nx,:nx] = np.eye(nx)

# Vu = np.zeros((ny, nu))
# Vu[nx:,:nu] = np.eye(nu)
# ocp.cost.Vu = Vu

# ocp.cost.Vx_e = np.eye(nx)

# ocp.cost.yref  = np.zeros((ny, ))
# ocp.cost.yref_e = np.zeros((ny_e, ))

# # set constraints
# Vmax = 1
# Omegamax = 1
# x0 = np.array([10.0, 30.0, np.pi])
# ocp.constraints.constr_type = 'BGH'
# ocp.constraints.lbu = np.array([-Vmax, -Omegamax])
# ocp.constraints.ubu = np.array([+Vmax, +Omegamax])
# ocp.constraints.x0 = x0
# ocp.constraints.idxbu = np.array([0,1])

# ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
# ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
# ocp.solver_options.integrator_type = 'ERK'
# ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI

# ocp.solver_options.qp_solver_cond_N = N

# # set prediction horizon
# ocp.solver_options.tf = Tf

# acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
# acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')

# simX = np.ndarray((N+1, nx))
# simU = np.ndarray((N, nu))

# xcurrent = x0
# simX[0,:] = xcurrent

# # closed loop
# for i in range(N):

#     # solve ocp
#     acados_ocp_solver.set(0, "lbx", xcurrent)
#     acados_ocp_solver.set(0, "ubx", xcurrent)

#     status = acados_ocp_solver.solve()

#     #if status != 0:
#     #    raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

#     simU[i,:] = acados_ocp_solver.get(0, "u")

#     # simulate system
#     acados_integrator.set("x", xcurrent)
#     acados_integrator.set("u", simU[i,:])

#     status = acados_integrator.solve()
#     if status != 0:
#         raise Exception('acados integrator returned status {}. Exiting.'.format(status))

#     # update state
#     xcurrent = acados_integrator.get("x")
#     simX[i+1,:] = xcurrent

Nsim = int(T * N / Tf)
# initialize data structs
simX = np.ndarray((Nsim, nx))
simU = np.ndarray((Nsim, nu))
simX_horizon = np.ndarray((Nsim, N, nx))
s0 = 0
tcomp_sum = 0
tcomp_max = 0

# simulate
for i in range(Nsim):
    # update reference
    sref = s0 + sref_N
    for j in range(N):
        yref = np.array([s0 + (sref - s0) * j / N, 0, 0, 0, 0])
        # yref=np.array([1,0,0,1,0,0,0,0])
        acados_solver.set(j, "yref", yref)
    yref_N = np.array([sref, 0, 0])
    # yref_N=np.array([0,0,0,0,0,0])
    acados_solver.set(N, "yref", yref_N)

    #for j in range(N):
    #    print(acados_solver.get(j, 'x'))
        #acados_solver.set(j, "con_h_expr", acados_solver.get(j, 'x'))

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

    # check if one lap is done and break and remove entries beyond
    if x0[0] > path.get_len() + 0.1:
        # find where vehicle first crosses start line
        N0 = np.where(np.diff(np.sign(simX[:, 0])))[0][0]
        Nsim = i - N0  # correct to final number of simulation steps for plotting
        simX = simX[N0:i, :]
        simU = simU[N0:i, :]
        break


def plotRes(simX,simU,t):
    # plot results
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.step(t, simU[:,0], color='r')
    plt.step(t, simU[:,1], color='g')
    plt.title('closed-loop simulation')
    plt.legend(['v','w'])
    plt.ylabel('u')
    plt.xlabel('t')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, simX[:,:])
    plt.ylabel('x')
    plt.xlabel('t')
    plt.legend(['s','l','theta_tilde'])
    plt.grid(True)


t = np.linspace(0.0, Nsim * Tf / N, Nsim)

time_now = datetime.datetime.now()
folder = time_now.strftime("%Y_%m_%d_%H:%M:%S")
os.mkdir('results/' + folder)

plotRes(simX, simU, t)
plt.savefig('results/' + folder + "/plots.png")
#plt.show()
# THIS IS A BIT SLOW
renderVideo(simX, simU, simX_horizon, t, car_model, path, folder)