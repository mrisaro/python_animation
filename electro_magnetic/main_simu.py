import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode


def e_of_x(x):
    return 10 * np.sign(np.sin(2 * np.pi * x / 25))

def compute_trajectory(m, q):
    r = ode(newton).set_integrator('dopri5')
    r.set_initial_value(initial_conditions, t0).set_f_params(m, q, 1.0, 10.)
    positions = []
    t1 = 200
    dt = 0.05
    while r.successful() and r.t < t1:
        r.set_f_params(m, q, 1.0, e_of_x(r.y[0]))
        r.integrate(r.t+dt)
        positions.append(r.y[:3])

    return np.array(positions)

positions = []
for m, q in zip([1, 0.1, 1, 0.1], [1, 1, -1, -1]):
    positions.append(compute_trajectory(m, q))
