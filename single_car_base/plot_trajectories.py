from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import SX, vertcat, sin, cos, Function
import numpy as np
import scipy.linalg
from CarModel import CarModel
from Path import Path
import time
import matplotlib.pyplot as plt
from utils import *

path = Path(10, 5, 2)

folder="/home/rossella/UNI/AMR/AMR_MPC-CBF/AMR_MPC-CBF/new_plots/disc_dynamic"
folder2="/home/rossella/UNI/AMR/AMR_MPC-CBF/AMR_MPC-CBF/new_plots/cont_dynamic"
folder3="/home/rossella/UNI/AMR/AMR_MPC-CBF/AMR_MPC-CBF/new_plots"

loadData(folder,folder2,folder3,path)
plt.show()

