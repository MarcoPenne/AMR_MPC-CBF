from acados_template import AcadosModel
from casadi import *
#from casadi import SX, vertcat, sin, cos, Function
from Path import Path
class CarModel:

    def __init__(self, path, l1, l2, fixed_obstacles, dT, n_lap):
        self.path = path
        self.l1 = l1
        self.l2 = l2
        self.dT = dT
        self.model = export_car_ode_model_with_discrete_rk4(path, l1, l2, fixed_obstacles, dT, n_lap)


def export_car_ode_model(path, l1, l2, fixed_obstacles, dT, n_lap):

    model_name = 'car_ode'

    # load track parameters
    pathlength = path.get_len()
    s0 = np.arange(0., n_lap*pathlength, 0.05)
    kapparef = np.zeros_like(s0)
    for i, s in enumerate(s0):
        kapparef[i] = path.get_k(s)
    
    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)

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
    p = vertcat([])
    
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

    #print(obs)

    x_t_plus_1 = x + f_expl*dT

    for o in range(obs.shape[0]):
        h_t = ((s-obs[o, 0])**4/(l1)**4) + ((l-obs[o, 1])**4/(l2)**4) - 5
        h_t_plus_1 = ((x_t_plus_1[0] - obs[o, 0])**4/(l1)**4) + ((x_t_plus_1[1] - obs[o, 1])**4/(l2)**4) - 5
    
        gamma = 0.1
        if o==0:
            model.con_h_expr = vertcat(h_t_plus_1 - h_t + gamma * h_t)
        else:
            model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

    return model


def export_car_ode_model_with_discrete_rk4(path, l1, l2, fixed_obstacles, dT, n_lap):

    model = export_car_ode_model(path, l1, l2, fixed_obstacles, dT, n_lap)

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

