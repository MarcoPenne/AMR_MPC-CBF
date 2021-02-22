from acados_template import AcadosModel
from casadi import *
#from casadi import SX, vertcat, sin, cos, Function
from Path import Path
class CarModel:

    def __init__(self, path, l1, l2, fixed_obstacles, dT, n_lap, gamma, h_cbf):
        self.path = path
        self.l1 = l1
        self.l2 = l2
        self.dT = dT
        self.model = export_car_ode_model_with_discrete_rk4(path, l1, l2, fixed_obstacles, dT, n_lap, gamma, h_cbf)


def export_car_ode_model(path, edge1, edge2, fixed_obstacles, dT, n_lap, gamma, h_cbf):

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
    l1 = MX.sym('l1')
    s1 = MX.sym('s1')
    theta_tilde1 = MX.sym('theta_tilde1')
    l2 = MX.sym('l2')
    s2 = MX.sym('s2')
    theta_tilde2 = MX.sym('theta_tilde2')
    l3 = MX.sym('l3')
    s3 = MX.sym('s3')
    theta_tilde3 = MX.sym('theta_tilde3')
    
    x = vertcat(s1, l1, theta_tilde1, s2, l2, theta_tilde2, s3, l3, theta_tilde3)

    # controls
    v1 = MX.sym('v1')
    omega1 = MX.sym('omega1')
    v2 = MX.sym('v2')
    omega2 = MX.sym('omega2')
    v3 = MX.sym('v3')
    omega3 = MX.sym('omega3')
    u = vertcat(v1, omega1, v2, omega2, v3, omega3)
    
    # xdot
    l1_dot      = MX.sym('l1_dot')
    s1_dot = MX.sym('s1_dot')
    theta_tilde1_dot = MX.sym('theta_tilde1_dot')
    l2_dot      = MX.sym('l2_dot')
    s2_dot = MX.sym('s2_dot')
    theta_tilde2_dot = MX.sym('theta_tilde2_dot')
    l3_dot      = MX.sym('l3_dot')
    s3_dot = MX.sym('s3_dot')
    theta_tilde3_dot = MX.sym('theta_tilde3_dot')

    xdot = vertcat(s1_dot, l1_dot, theta_tilde1_dot, s2_dot, l2_dot, theta_tilde2_dot, s3_dot, l3_dot, theta_tilde3_dot)

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
    # p = vertcat(s_obs1, l_obs1, theta_tilde_obs1, v_obs1, s_obs2, l_obs2, theta_tilde_obs2, v_obs2)
    p = vertcat([])

    # dynamics
    f_expl = vertcat((v1  * cos(theta_tilde1))/(1 - kapparef_s(s1)*l1),
                    v1 * sin(theta_tilde1),
                    omega1 - ((kapparef_s(s1) *v1*cos(theta_tilde1)/(1- kapparef_s(s1)*l1))),
                    (v2  * cos(theta_tilde2))/(1 - kapparef_s(s2)*l2),
                    v2 * sin(theta_tilde2),
                    omega2 - ((kapparef_s(s2) *v2*cos(theta_tilde2)/(1- kapparef_s(s2)*l2))),
                    (v3  * cos(theta_tilde3))/(1 - kapparef_s(s3)*l3),
                    v3 * sin(theta_tilde3),
                    omega3 - ((kapparef_s(s3) *v3*cos(theta_tilde3)/(1- kapparef_s(s3)*l3)))
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

    for i in range(3):
        for j in range(i+1, 3):
            h_t = ((x[i*3] - x[j*3])**4/(edge1)**4) + ((x[1+i*3]-x[1+j*3])**4/(edge2)**4) - h_cbf
            h_t_plus_1 = ((x_t_plus_1[i*3] - x_t_plus_1[j*3])**4/(edge1)**4) + ((x_t_plus_1[1+i*3] - x_t_plus_1[1+j*3])**4/(edge2)**4) - h_cbf
        
            if model.con_h_expr is None: 
                model.con_h_expr = vertcat(h_t_plus_1 - h_t + gamma * h_t)
            else:
                model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

            for lap in range(1, n_lap):
                h_t = ((x[i*3] - (x[j*3]+lap*path.get_len()))**4/(edge1)**4) + ((x[1+i*3]-x[1+j*3])**4/(edge2)**4) - h_cbf
                h_t_plus_1 = ((x_t_plus_1[i*3] - (x_t_plus_1[j*3]+lap*path.get_len()))**4/(edge1)**4) + ((x_t_plus_1[1+i*3] - x_t_plus_1[1+j*3])**4/(edge2)**4) - h_cbf
                
                model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)
            for lap in range(-1, -n_lap):
                h_t = ((x[i*3] - (x[j*3]+lap*path.get_len()))**4/(edge1)**4) + ((x[1+i*3]-x[1+j*3])**4/(edge2)**4) - h_cbf
                h_t_plus_1 = ((x_t_plus_1[i*3] - (x_t_plus_1[j*3]+lap*path.get_len()))**4/(edge1)**4) + ((x_t_plus_1[1+i*3] - x_t_plus_1[1+j*3])**4/(edge2)**4) - h_cbf
                
            model.con_h_expr = vertcat(model.con_h_expr, h_t_plus_1 - h_t + gamma * h_t)

                
    if obs is not None:
        for o in range(obs.shape[0]):
            for i in range(3):
                h1_t = ((x[i*3]-obs[o, 0])**4/(edge1)**4) + ((x[1+i*3]-obs[o, 1])**4/(edge2)**4) - h_cbf
                h1_t_plus_1 = ((x_t_plus_1[i*3] - obs[o, 0])**4/(edge1)**4) + ((x_t_plus_1[1+i*3] - obs[o, 1])**4/(edge2)**4) - h_cbf
            
                model.con_h_expr = vertcat(model.con_h_expr,
                                            h1_t_plus_1 - h1_t + gamma * h1_t
                                            )

    return model


def export_car_ode_model_with_discrete_rk4(path, l1, l2, fixed_obstacles, dT, n_lap, gamma, h_cbf):

    model = export_car_ode_model(path, l1, l2, fixed_obstacles, dT, n_lap, gamma, h_cbf)

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

