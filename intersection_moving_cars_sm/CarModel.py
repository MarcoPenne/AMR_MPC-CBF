from acados_template import AcadosModel
from casadi import *
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import SX, vertcat, sin, cos, Function
import scipy.linalg

#from casadi import SX, vertcat, sin, cos, Function
from Path import Path
class CarModel:

    def __init__(self, path, l1, l2, fixed_obstacles, other_path, other_obstacles, dT, n_lap, name, gamma, h_cbf):
        self.path = path
        self.l1 = l1
        self.l2 = l2
        self.dT = dT
        self.other_path = other_path
        self.name = name
        self.other_obstacles = other_obstacles
        self.model = export_car_ode_model_with_discrete_rk4(path, l1, l2, fixed_obstacles, other_path, other_obstacles, dT, n_lap, name, gamma, h_cbf)


def export_car_ode_model(path, edge1, edge2, fixed_obstacles, other_path, other_obstacles, dT, n_lap, name, gamma, h_cbf):

    model_name = 'car_ode_'+name

    # load track parameters
    pathlength1 = path.get_len()
    s01 = np.arange(0., n_lap*pathlength1, 0.05)
    kapparef1 = np.zeros_like(s01)
    xref1 = np.zeros_like(s01)
    yref1 = np.zeros_like(s01)
    thetarref1 = np.zeros_like(s01)
    for i, s in enumerate(s01):
        kapparef1[i] = path.get_k(s)
        thetarref1[i] = path.get_theta_r(s)
        (_x, _y) = path(s)
        xref1[i] = _x
        yref1[i] = _y

    pathlength2 = other_path.get_len()
    s02 = np.arange(0., n_lap*pathlength2, 0.05)
    kapparef2 = np.zeros_like(s02)
    xref2 = np.zeros_like(s02)
    yref2 = np.zeros_like(s02)
    thetarref2 = np.zeros_like(s02)
    for i, s in enumerate(s02):
        kapparef2[i] = other_path.get_k(s)
        thetarref2[i] = other_path.get_theta_r(s)
        (_x, _y) = other_path(s)
        xref2[i] = _x
        yref2[i] = _y
    
    # compute spline interpolations
    kapparef_s1 = interpolant("kapparef_s1", "bspline", [s01], kapparef1)
    xref_s1 = interpolant("xref_s1", "bspline", [s01], xref1)
    yref_s1 = interpolant("yref_s1", "bspline", [s01], yref1)
    thetarref_s1 = interpolant("thetarref_s1", "bspline", [s01], thetarref1)

    kapparef_s2 = interpolant("kapparef_s2", "bspline", [s02], kapparef2)
    xref_s2 = interpolant("xref_s2", "bspline", [s02], xref2)
    yref_s2 = interpolant("yref_s2", "bspline", [s02], yref2)
    thetarref_s2 = interpolant("thetarref_s2", "bspline", [s02], thetarref2)

    # set up states & controls
    l1 = MX.sym('l1')
    s1 = MX.sym('s1')
    theta_tilde1 = MX.sym('theta_tilde1')
    l2 = MX.sym('l2')
    s2 = MX.sym('s2')
    theta_tilde2 = MX.sym('theta_tilde2')
    
    x = vertcat(s1, l1, theta_tilde1, s2, l2, theta_tilde2)

    # controls
    v1 = MX.sym('v1')
    omega1 = MX.sym('omega1')
    v2 = MX.sym('v2')
    omega2 = MX.sym('omega2')
    u = vertcat(v1, omega1, v2, omega2)
    
    # xdot
    l_dot1      = MX.sym('l_dot1')
    s_dot1 = MX.sym('s_dot1')
    theta_tilde_dot1 = MX.sym('theta_tilde_dot1')
    l_dot2      = MX.sym('l_dot2')
    s_dot2 = MX.sym('s_dot2')
    theta_tilde_dot2 = MX.sym('theta_tilde_dot2')
    xdot = vertcat(s_dot1, l_dot1, theta_tilde_dot1, s_dot2, l_dot2, theta_tilde_dot2)

    # algebraic variables
    z = vertcat([])

    # parameters
    x_obs1 = MX.sym('x_obs1')
    y_obs1 = MX.sym('y_obs1')
    theta_obs1 = MX.sym('theta_obs1')
    v_obs1 = MX.sym('v_obs1')

    x_obs2 = MX.sym('x_obs2')
    y_obs2 = MX.sym('y_obs2')
    theta_obs2 = MX.sym('theta_obs2')
    v_obs2 = MX.sym('v_obs2')
    
    x_obs3 = MX.sym('x_obs3')
    y_obs3 = MX.sym('y_obs3')
    theta_obs3 = MX.sym('theta_obs3')
    v_obs3 = MX.sym('v_obs3')
    
    x_obs4 = MX.sym('x_obs4')
    y_obs4 = MX.sym('y_obs4')
    theta_obs4 = MX.sym('theta_obs4')
    v_obs4 = MX.sym('v_obs4')
    p = vertcat(x_obs1, y_obs1, theta_obs1, v_obs1,
                x_obs2, y_obs2, theta_obs2, v_obs2,
                x_obs3, y_obs3, theta_obs3, v_obs3,
                x_obs4, y_obs4, theta_obs4, v_obs4)
    
    # dynamics
    f_expl = vertcat((v1  * cos(theta_tilde1))/(1 - kapparef_s1(s1)*l1),
                    v1 * sin(theta_tilde1),
                    omega1 - ((kapparef_s1(s1) *v1*cos(theta_tilde1)/(1- kapparef_s1(s1)*l1))),
                    (v2  * cos(theta_tilde2))/(1 - kapparef_s2(s2)*l2),
                    v2 * sin(theta_tilde2),
                    omega2 - ((kapparef_s2(s2) *v2*cos(theta_tilde2)/(1- kapparef_s2(s2)*l2)))
                    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name

    # obs=None
    # if fixed_obstacles is not None:
    #     obs = fixed_obstacles
    #     lap_vector = np.zeros_like(obs)
    #     lap_vector[:, 0] = path.get_len()
    #     tmp_obs = fixed_obstacles
    #     obs = np.concatenate((obs, tmp_obs - lap_vector), axis=0)

    #     for lap in range(1, n_lap):
    #         lap_vector = np.zeros_like(fixed_obstacles)
    #         lap_vector[:, 0] = lap*path.get_len()
    #         tmp_obs = fixed_obstacles
    #         obs = np.concatenate((obs, tmp_obs + lap_vector), axis=0)

    
    x_t_plus_1 = x + f_expl*dT

    thetar_car1 = thetarref_s1(s1)
    x_car1 = xref_s1(s1) - sin(thetar_car1)*l1
    y_car1 = yref_s1(s1) + cos(thetar_car1)*l1
    
    thetar_car_next_1 = thetarref_s1(x_t_plus_1[0])
    x_car_next_1 = xref_s1(x_t_plus_1[0]) - sin(thetar_car_next_1)*x_t_plus_1[1]
    y_car_next_1 = yref_s1(x_t_plus_1[0]) + cos(thetar_car_next_1)*x_t_plus_1[1]

    thetar_car2 = thetarref_s2(s2)
    x_car2 = xref_s2(s2) - sin(thetar_car2)*l2
    y_car2 = yref_s2(s2) + cos(thetar_car2)*l2
    
    thetar_car_next_2 = thetarref_s2(x_t_plus_1[3])
    x_car_next_2 = xref_s2(x_t_plus_1[3]) - sin(thetar_car_next_2)*x_t_plus_1[4]
    y_car_next_2 = yref_s2(x_t_plus_1[3]) + cos(thetar_car_next_2)*x_t_plus_1[4]

    # if other_obstacles is not None:
    #     for i in range(other_obstacles.shape[0]):
    #         o = other_obstacles[i]
    #         o = other_path.get_cartesian_coords(o[0], o[1])
            
    #         h_t = ((x_car - o[0] )**4/(l1)**4) + ((y_car - o[1])**4/(l1)**4) - 1
    #         h_t_plus_1 = ((x_car_1 - o[0])**4/(l1)**4) + ((y_car_1 - o[1])**4/(l1)**4) - 1
            
    #         if model.con_h_expr is not None:
    #             model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)
    #         else:
    #             model.con_h_expr = vertcat(h_t_plus_1 - h_t + gamma * h_t)
    
    # if obs is not None:
    #     for i in range(obs.shape[0]):
    #         o = path.get_cartesian_coords(obs[i, 0], obs[i, 1])
            
    #         h_t = ((x_car - o[0] )**4/(l1)**4) + ((y_car - o[1])**4/(l1)**4) - 1
    #         h_t_plus_1 = ((x_car_1 - o[0])**4/(l1)**4) + ((y_car_1 - o[1])**4/(l1)**4) - 1

    #         if model.con_h_expr is None:
    #             model.con_h_expr = vertcat(h_t_plus_1 - h_t + gamma * h_t)
    #         else:
    #             model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    # CBFs
    # First Car
    h_t = ((x_car1 - x_obs1 )**4/(edge1)**4) + ((y_car1 - y_obs1)**4/(edge1)**4) - h_cbf
    h_t_plus_1 = ((x_car_next_1 - (x_obs1+cos(theta_obs1)*v_obs1*dT))**4/(edge1)**4) + ((y_car_next_1 - (y_obs1+sin(theta_obs1)*v_obs1*dT))**4/(edge1)**4) - h_cbf
    if model.con_h_expr is None:
        model.con_h_expr = vertcat(h_t_plus_1 - h_t + gamma * h_t)
    else:
        model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    h_t = ((x_car1 - x_obs2 )**4/(edge1)**4) + ((y_car1 - y_obs2)**4/(edge1)**4) - h_cbf
    h_t_plus_1 = ((x_car_next_1 - (x_obs2+cos(theta_obs2)*v_obs2*dT))**4/(edge1)**4) + ((y_car_next_1 - (y_obs2+sin(theta_obs2)*v_obs2*dT))**4/(edge1)**4) - h_cbf
    model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    h_t = ((x_car1 - x_obs3 )**4/(edge1)**4) + ((y_car1 - y_obs3)**4/(edge1)**4) - h_cbf
    h_t_plus_1 = ((x_car_next_1 - (x_obs3+cos(theta_obs3)*v_obs3*dT))**4/(edge1)**4) + ((y_car_next_1 - (y_obs3+sin(theta_obs3)*v_obs3*dT))**4/(edge1)**4) - h_cbf
    model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    h_t = ((x_car1 - x_obs4 )**4/(edge1)**4) + ((y_car1 - y_obs4)**4/(edge1)**4) - h_cbf
    h_t_plus_1 = ((x_car_next_1 - (x_obs4+cos(theta_obs4)*v_obs4*dT))**4/(edge1)**4) + ((y_car_next_1 - (y_obs4+sin(theta_obs4)*v_obs4*dT))**4/(edge1)**4) - h_cbf
    model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    # Second Car
    h_t = ((x_car2 - x_obs1 )**4/(edge1)**4) + ((y_car2 - y_obs1)**4/(edge1)**4) - h_cbf
    h_t_plus_1 = ((x_car_next_2 - (x_obs1+cos(theta_obs1)*v_obs1*dT))**4/(edge1)**4) + ((y_car_next_2 - (y_obs1+sin(theta_obs1)*v_obs1*dT))**4/(edge1)**4) - h_cbf
    model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    h_t = ((x_car2 - x_obs2 )**4/(edge1)**4) + ((y_car2 - y_obs2)**4/(edge1)**4) - h_cbf
    h_t_plus_1 = ((x_car_next_2 - (x_obs2+cos(theta_obs2)*v_obs2*dT))**4/(edge1)**4) + ((y_car_next_2 - (y_obs2+sin(theta_obs2)*v_obs2*dT))**4/(edge1)**4) - h_cbf
    model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    h_t = ((x_car2 - x_obs3 )**4/(edge1)**4) + ((y_car2 - y_obs3)**4/(edge1)**4) - h_cbf
    h_t_plus_1 = ((x_car_next_2 - (x_obs3+cos(theta_obs3)*v_obs3*dT))**4/(edge1)**4) + ((y_car_next_2 - (y_obs3+sin(theta_obs3)*v_obs3*dT))**4/(edge1)**4) - h_cbf
    model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    h_t = ((x_car2 - x_obs4 )**4/(edge1)**4) + ((y_car2 - y_obs4)**4/(edge1)**4) - h_cbf
    h_t_plus_1 = ((x_car_next_2 - (x_obs4+cos(theta_obs4)*v_obs4*dT))**4/(edge1)**4) + ((y_car_next_2 - (y_obs4+sin(theta_obs4)*v_obs4*dT))**4/(edge1)**4) - h_cbf
    model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    # Between cars
    h_t = ((x_car1 - x_car2 )**4/(edge1)**4) + ((y_car1 - y_car2)**4/(edge1)**4) - h_cbf
    h_t_plus_1 = ((x_car_next_1 - x_car_next_2)**4/(edge1)**4) + ((y_car_next_1 - y_car_next_2)**4/(edge1)**4) - h_cbf
    model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)
    return model


def export_car_ode_model_with_discrete_rk4(path, l1, l2, fixed_obstacles, other_path, other_obstacles, dT, n_lap, name, gamma, h_cbf):

    model = export_car_ode_model(path, l1, l2, fixed_obstacles, other_path, other_obstacles, dT, n_lap, name, gamma, h_cbf)

    x = model.x
    u = model.u
    nx = x.size()[0]

    ode = Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1,u)
    k3 = ode(x+dT/2*k2,u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf
    print("built RK4 for car model with dT = ", dT)
    # print(xf)
    # print()
    # print(k1)
    # print(k2)
    # print(k3)
    # print(k4)
    
    # print(xf.shape)
    
    #print( xf.set() )
    
    return model



def create_problem(path, l1, l2, fixed_obstacles, other_path, other_obstacles, N, Tf, n_lap, x0, name, gamma, h_cbf):
    dT = Tf/float(N)
    car_model = CarModel(path, l1, l2, fixed_obstacles, other_path, other_obstacles, dT, n_lap, name, gamma, h_cbf)
    model = car_model.model
    ocp = AcadosOcp()
    ocp.model = model
    #print(model)

    # set dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N

    # set cost
    Q = np.diag([ 100, 10, 10, 100, 10, 10])
    R = np.diag([10, 10, 10, 10])#np.eye(nu)*1e-1

    Qe = np.diag([ 100, 10, 10, 100, 10, 10])

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
    ocp.cost.yref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0, 0, 0, 0, 0])

    ocp.parameter_values = np.array([0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0])

    # setting constraints
    ocp.constraints.lbx = np.array([-2, -2])
    ocp.constraints.ubx = np.array([2, 2])
    ocp.constraints.idxbx = np.array([1, 4])
    ocp.constraints.lbu = np.array([-4, -2, -4, -2])
    ocp.constraints.ubu = np.array([4, 2, 4, 2])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    #  Set CBF
    #
    if model.con_h_expr is not None:
        print(f"Created {model.con_h_expr.shape[0]} CBFs")
        ocp.constraints.lh = np.zeros(model.con_h_expr.shape[0])
        ocp.constraints.uh = np.ones(model.con_h_expr.shape[0])*1e15

    # set intial condition
    ocp.constraints.x0 = x0 

    # set QP solver and integration
    ocp.solver_options.tf = Tf
    #ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    #ocp.solver_options.qp_solver_iter_max = 300
    #ocp.solver_options.nlp_solver_max_iter = 300

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_"+name+".json")
    return acados_solver, car_model