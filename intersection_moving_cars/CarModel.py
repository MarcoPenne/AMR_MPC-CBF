from acados_template import AcadosModel
from casadi import *
#from casadi import SX, vertcat, sin, cos, Function
from Path import Path
class CarModel:

    def __init__(self, path, l1, l2, fixed_obstacles, other_path, other_obstacles, dT, n_lap):
        self.path = path
        self.l1 = l1
        self.l2 = l2
        self.dT = dT
        self.other_path = other_path
        self.other_obstacles = other_obstacles
        self.model = export_car_ode_model_with_discrete_rk4(path, l1, l2, fixed_obstacles, other_path, other_obstacles, dT, n_lap)


def export_car_ode_model(path, l1, l2, fixed_obstacles, other_path, other_obstacles, dT, n_lap):

    model_name = 'car_ode'

    # load track parameters
    pathlength = path.get_len()
    s0 = np.arange(0., n_lap*pathlength, 0.05)
    kapparef = np.zeros_like(s0)
    xref = np.zeros_like(s0)
    yref = np.zeros_like(s0)
    thetarref = np.zeros_like(s0)
    for i, s in enumerate(s0):
        kapparef[i] = path.get_k(s)
        thetarref[i] = path.get_theta_r(s)
        (_x, _y) = path(s)
        xref[i] = _x
        yref[i] = _y
    
    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    xref_s = interpolant("xref_s", "bspline", [s0], xref)
    yref_s = interpolant("yref_s", "bspline", [s0], yref)
    thetarref_s = interpolant("thetarref_s", "bspline", [s0], thetarref)

    # set up states & controls
    l = MX.sym('l')
    s = MX.sym('s')
    theta_tilde = MX.sym('theta_tilde')
    
    x = vertcat(s, l, theta_tilde)

    # controls
    v1 = MX.sym('v1')
    omega = MX.sym('omega')
    u = vertcat(v1, omega)
    
    # xdot
    l_dot      = MX.sym('l_dot')
    s_dot = MX.sym('s_dot')
    theta_tilde_dot = MX.sym('theta_tilde_dot')

    xdot = vertcat(s_dot, l_dot, theta_tilde_dot)

    # algebraic variables
    z = vertcat([])

    # parameters
    # s_obs1 = MX.sym('s_obs1')
    # l_obs1 = MX.sym('l_obs1')
    # theta_tilde_obs1 = MX.sym('theta_tilde_obs1')
    # v_obs1 = MX.sym('v_obs1')
    # s_obs2 = MX.sym('s_obs2')
    # l_obs2 = MX.sym('l_obs2')
    # theta_tilde_obs2 = MX.sym('theta_tilde_obs2')
    # v_obs2 = MX.sym('v_obs2')
    
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
    p = vertcat(x_obs1, y_obs1, theta_obs1, v_obs1,
                x_obs2, y_obs2, theta_obs2, v_obs2,
                x_obs3, y_obs3, theta_obs3, v_obs3)
    
    # dynamics
    f_expl = vertcat((v1  * cos(theta_tilde))/(1 - kapparef_s(s)*l),
                    v1 * sin(theta_tilde),
                    omega - ((kapparef_s(s) *v1*cos(theta_tilde)/(1- kapparef_s(s)*l)))
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

    obs=None
    if fixed_obstacles is not None:
        obs = fixed_obstacles
        lap_vector = np.zeros_like(obs)
        lap_vector[:, 0] = path.get_len()
        tmp_obs = fixed_obstacles
        obs = np.concatenate((obs, tmp_obs - lap_vector), axis=0)

        for lap in range(1, n_lap):
            lap_vector = np.zeros_like(fixed_obstacles)
            lap_vector[:, 0] = lap*path.get_len()
            tmp_obs = fixed_obstacles
            obs = np.concatenate((obs, tmp_obs + lap_vector), axis=0)

    
    x_t_plus_1 = x + f_expl*dT

    gamma = 0.1

    thetar_car = thetarref_s(s)
    x_car = xref_s(s) - sin(thetar_car)*l
    y_car = yref_s(s) + cos(thetar_car)*l
    
    thetar_car_1 = thetarref_s(x_t_plus_1[0])
    x_car_1 = xref_s(x_t_plus_1[0]) - sin(thetar_car_1)*x_t_plus_1[1]
    y_car_1 = yref_s(x_t_plus_1[0]) + cos(thetar_car_1)*x_t_plus_1[1]

    if other_obstacles is not None:
        for i in range(other_obstacles.shape[0]):
            o = other_obstacles[i]
            o = other_path.get_cartesian_coords(o[0], o[1])
            
            h_t = ((x_car - o[0] )**4/(l1)**4) + ((y_car - o[1])**4/(l1)**4) - 1
            h_t_plus_1 = ((x_car_1 - o[0])**4/(l1)**4) + ((y_car_1 - o[1])**4/(l1)**4) - 1
            
            if model.con_h_expr is not None:
                model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)
            else:
                model.con_h_expr = vertcat(h_t_plus_1 - h_t + gamma * h_t)
    
    if obs is not None:
        for i in range(obs.shape[0]):
            o = path.get_cartesian_coords(obs[i, 0], obs[i, 1])
            
            h_t = ((x_car - o[0] )**4/(l1)**4) + ((y_car - o[1])**4/(l1)**4) - 1
            h_t_plus_1 = ((x_car_1 - o[0])**4/(l1)**4) + ((y_car_1 - o[1])**4/(l1)**4) - 1

            if model.con_h_expr is None:
                model.con_h_expr = vertcat(h_t_plus_1 - h_t + gamma * h_t)
            else:
                model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)


    h_t = ((x_car - x_obs1 )**4/(l1)**4) + ((y_car - y_obs1)**4/(l1)**4) - 1
    h_t_plus_1 = ((x_car_1 - (x_obs1+cos(theta_obs1)*v_obs1*dT))**4/(l1)**4) + ((y_car_1 - (y_obs1+sin(theta_obs1)*v_obs1*dT))**4/(l1)**4) - 1
    if model.con_h_expr is None:
        model.con_h_expr = vertcat(h_t_plus_1 - h_t + gamma * h_t)
    else:
        model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    h_t = ((x_car - x_obs2 )**4/(l1)**4) + ((y_car - y_obs2)**4/(l1)**4) - 1
    h_t_plus_1 = ((x_car_1 - (x_obs2+cos(theta_obs2)*v_obs2*dT))**4/(l1)**4) + ((y_car_1 - (y_obs2+sin(theta_obs2)*v_obs2*dT))**4/(l1)**4) - 1
    model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    h_t = ((x_car - x_obs3 )**4/(l1)**4) + ((y_car - y_obs3)**4/(l1)**4) - 1
    h_t_plus_1 = ((x_car_1 - (x_obs3+cos(theta_obs3)*v_obs3*dT))**4/(l1)**4) + ((y_car_1 - (y_obs3+sin(theta_obs3)*v_obs3*dT))**4/(l1)**4) - 1
    model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)
    return model


def export_car_ode_model_with_discrete_rk4(path, l1, l2, fixed_obstacles, other_path, other_obstacles, dT, n_lap):

    model = export_car_ode_model(path, l1, l2, fixed_obstacles, other_path, other_obstacles, dT, n_lap)

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

