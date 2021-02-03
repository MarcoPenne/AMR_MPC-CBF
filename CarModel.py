from acados_template import AcadosModel
from casadi import *
#from casadi import SX, vertcat, sin, cos, Function
from Path import Path
class CarModel:

    def __init__(self, path, l1, l2, dT):
        self.path = path
        self.l1 = l1
        self.l2 = l2
        self.dT = dT
        self.model = export_car_ode_model_with_discrete_rk4(path, l1, l2, dT)


def export_car_ode_model(path, l1, l2):

    model_name = 'car_ode'

    # load track parameters
    pathlength = path.get_len()
    s0 = np.arange(0., pathlength, 0.05)
    kapparef = np.zeros_like(s0)
    for i, s in enumerate(s0):
        kapparef[i] = path.get_k(s)
    
    # [s0, _, _, _, kapparef] = getTrack(track)
    # length = len(s0)
    # pathlength = s0[-1]
    # # copy loop to beginning and end
    # s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    # kapparef = np.append(kapparef, kapparef[1:length])
    # s0 = np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0)
    # kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)

    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    print(kapparef_s)
    print(kapparef_s(5))
    print(kapparef_s(10.1))

    # set up states & controls
    l      = MX.sym('l')
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

    return model


def export_car_ode_model_with_discrete_rk4(path, l1, l2, dT):

    model = export_car_ode_model(path, l1, l2)

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
    print(xf)
    return model

