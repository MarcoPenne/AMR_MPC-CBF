from acados_template import AcadosModel
from casadi import *
#from casadi import SX, vertcat, sin, cos, Function
from Path import Path

class CarModel:

    def __init__(self, path, l1, l2, fixed_obstacles, n_lap, K, h_cbf):
        self.path = path
        self.l1 = l1
        self.l2 = l2
        self.model = export_car_ode_model(path, l1, l2, fixed_obstacles, n_lap, K, h_cbf)


def export_car_ode_model(path, l1, l2, fixed_obstacles, n_lap, K, h_cbf):

    model_name = 'car_ode'

    # load track parameters
    pathlength = path.get_len()
    s0 = np.arange(0., n_lap*pathlength, 0.05)
    kapparef = np.zeros_like(s0)
    for i, s in enumerate(s0):
        kapparef[i] = path.get_k(s)

    kapparef_dot = np.diff(kapparef)
    s0_dot = np.arange(0.025, n_lap*pathlength, 0.05)[:kapparef.shape[0]-1]
    print(kapparef.shape)
    print(s0.shape)
    print(kapparef_dot.shape)
    print(s0_dot.shape)
    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    kapparef_s_dot = interpolant("kapparef_s_dot", "bspline", [s0_dot], kapparef_dot)
    print(kapparef_s)
    print(kapparef_s_dot)

    # set up states & controls
    l = MX.sym('l')
    s = MX.sym('s')
    theta_tilde = MX.sym('theta_tilde')
    v1 = MX.sym('v1')
    omega = MX.sym('omega')
    x = vertcat(s, l, theta_tilde, v1, omega)

    # controls
    a1 = MX.sym('a1')
    aw = MX.sym('aw')
    u = vertcat(a1, aw)
    
    # xdot
    l_dot      = MX.sym('l_dot')
    s_dot = MX.sym('s_dot')
    theta_tilde_dot = MX.sym('theta_tilde_dot')
    v1_dot = MX.sym('v1_dot')
    omega_dot = MX.sym('omega_dot')
    xdot = vertcat(s_dot, l_dot, theta_tilde_dot, v1_dot, omega_dot)

    # algebraic variables
    z = vertcat([])

    # parameters
    s_obs1 = MX.sym('s_obs1')
    l_obs1 = MX.sym('l_obs1')
    theta_tilde_obs1 = MX.sym('theta_tilde_obs1')
    v_obs1 = MX.sym('v_obs1')
    s_obs2 = MX.sym('s_obs2')
    l_obs2 = MX.sym('l_obs2')
    theta_tilde_obs2 = MX.sym('theta_tilde_obs2')
    v_obs2 = MX.sym('v_obs2')
    p = vertcat(s_obs1, l_obs1, theta_tilde_obs1, v_obs1, s_obs2, l_obs2, theta_tilde_obs2, v_obs2)

    # dynamics
    f_expl = vertcat((v1  * cos(theta_tilde))/(1 - kapparef_s(s)*l),
                    v1 * sin(theta_tilde),
                    omega - ((kapparef_s(s) *v1*cos(theta_tilde)/(1- kapparef_s(s)*l))),
                    a1,
                    aw
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

    l_car_dot = v1 * sin(theta_tilde)
    s_car_dot = (v1  * cos(theta_tilde))/(1 - kapparef_s(s)*l)
    theta_tilde_car_dot = omega - ((kapparef_s(s) *v1*cos(theta_tilde)/(1- kapparef_s(s)*l)))
    s_car_ddot = ((a1*cos(theta_tilde)-v1*sin(theta_tilde)*theta_tilde_car_dot)*(1 - kapparef_s(s)*l) + (v1  * cos(theta_tilde))*(-kapparef_s_dot(s)*l - kapparef_s(s)*l_car_dot))/((1 - kapparef_s(s)*l)**2)
    l_car_ddot = a1* sin(theta_tilde) + v1*cos(theta_tilde)*theta_tilde_car_dot

    if obs is not None:
        for o in range(obs.shape[0]):
            h_t = ((s-obs[o, 0])**4/(l1)**4) + ((l-obs[o, 1])**4/(l2)**4) - h_cbf
            h_t_dot = ( (4*((s-obs[o, 0])**3)*s_car_dot) / (l1)**4) + ( (4*((l-obs[o, 1])**3)*l_car_dot) / (l2)**4)
            h_t_ddot = (1/l1**4)*( (12*((s-obs[o, 0])**2)*(s_car_dot**2) )+(4*((s-obs[o, 0])**3)*s_car_ddot)) + (1/l2**4)*((12*((l-obs[o, 1])**2)*(l_car_dot**2))+(4*((l-obs[o, 1])**3)*l_car_ddot))
        
            # h' = hdot + gamma(h)
            # h'dot = 
            if o==0:
                #model.con_h_expr = vertcat(h_t_ddot + ((K[0]*K[1])*h_t + (K[0]+K[1])*h_t_dot))
                model.con_h_expr = vertcat(h_t_ddot + ((K[0])*h_t + (K[1])*h_t_dot))
            else:
                #model.con_h_expr = vertcat(model.con_h_expr, h_t_ddot + ((K[0]*K[1])*h_t + (K[0]+K[1])*h_t_dot))
                model.con_h_expr = vertcat(model.con_h_expr, h_t_ddot + ((K[0])*h_t + (K[1])*h_t_dot))

    for lap in range(0, n_lap):
        h_t = ((s - (s_obs1+lap*path.get_len()) )**4/(l1)**4) + ((l-l_obs1)**4/(l2)**4) - h_cbf
        h_t_dot = ( (4*((s- (s_obs1+lap*path.get_len()))**3)*(s_car_dot - v_obs1)) / (l1)**4) + ( (4*((l-l_obs1)**3)*l_car_dot) / (l2)**4)
        h_t_ddot = (1/(l1)**4)*(12*((s- (s_obs1+lap*path.get_len()))**2)*(s_car_dot-v_obs1)**2 + 4*((s- (s_obs1+lap*path.get_len()))**3)*(s_car_ddot)) +\
             (1/(l2)**4)*((12*(l-l_obs1)**2*(l_car_dot)**2) + (4*((l-l_obs1)**3) * l_car_ddot))

        if lap==0 and fixed_obstacles is None:
            model.con_h_expr = vertcat(h_t_ddot + ((K[0])*h_t + (K[1])*h_t_dot))
        else:
            model.con_h_expr = vertcat(model.con_h_expr, h_t_ddot + ((K[0])*h_t + (K[1])*h_t_dot))

        h_t = ((s - (s_obs2+lap*path.get_len()) )**4/(l1)**4) + ((l-l_obs2)**4/(l2)**4) - h_cbf
        h_t_dot = ( (4*((s- (s_obs2+lap*path.get_len()))**3)*(s_car_dot - v_obs2)) / (l1)**4) + ( (4*((l-l_obs2)**3)*l_car_dot) / (l2)**4)
        h_t_ddot = (1/(l1)**4)*(12*((s- (s_obs2+lap*path.get_len()))**2)*(s_car_dot-v_obs2)**2 + 4*((s- (s_obs2+lap*path.get_len()))**3)*(s_car_ddot)) +\
                    (1/(l2)**4)*((12*(l-l_obs2)**2*(l_car_dot)**2) + (4*((l-l_obs2)**3) * l_car_ddot))
        model.con_h_expr = vertcat(model.con_h_expr, h_t_ddot + ((K[0])*h_t + (K[1])*h_t_dot))


    return model


# def export_car_ode_model_with_discrete_rk4(path, l1, l2, fixed_obstacles, dT, n_lap):

#     model = export_car_ode_model(path, l1, l2, fixed_obstacles, dT, n_lap)

#     x = model.x
#     u = model.u
#     nx = x.size()[0]

#     ode = Function('ode', [x, u], [model.f_expl_expr])
#     # set up RK4
#     k1 = ode(x,       u)
#     k2 = ode(x+dT/2*k1,u)
#     k3 = ode(x+dT/2*k2,u)
#     k4 = ode(x+dT*k3,  u)
#     xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

#     model.disc_dyn_expr = xf
#     print("built RK4 for car model with dT = ", dT)
#     # print(xf)
#     # print()
#     # print(k1)
#     # print(k2)
#     # print(k3)
#     # print(k4)
    
#     # print(xf.shape)
    
#     #print( xf.set() )
    
#     return model

