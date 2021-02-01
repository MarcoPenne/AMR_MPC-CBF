from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, Function

class CarModel:

    def __init__(self, l1, l2, dT):
        self.l1 = l1
        self.l2 = l2
        self.dT = dT
        self.model = export_car_ode_model_with_discrete_rk4(l1, l2, dT)


def export_car_ode_model(l1, l2):

    model_name = 'car_ode'

    # set up states & controls
    l      = SX.sym('l')
    s = SX.sym('s')
    theta_tilde = SX.sym('theta_tilde')
    
    x = vertcat(l, s, theta_tilde)

    # controls
    v1 = SX.sym('v1')
    omega = SX.sym('omega')
    u = vertcat(v1, omega)
    
    # xdot
    l_dot      = SX.sym('l_dot')
    s_dot = SX.sym('s_dot')
    theta_tilde_dot = SX.sym('theta_tilde_dot')

    xdot = vertcat(l_dot, s_dot, theta_tilde_dot)

    # algebraic variables
    # z = None

    # parameters
    k = SX.sym('k')
    #theta_r = SX.sym('theta_r')
    p = [k]
    
    # dynamics
    f_expl = vertcat(v1 * sin(theta_tilde),
                     (v1  * cos(theta_tilde))/(1-k*l),
                     omega - ((k*v1*cos(theta_tilde)/(1-k*l)))
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    return model


def export_car_ode_model_with_discrete_rk4(l1, l2, dT):

    model = export_car_ode_model(l1, l2)

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

CarModel(1, 0.5, 0.1)