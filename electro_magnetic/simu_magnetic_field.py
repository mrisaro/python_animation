# Librerías a utilizar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
from matplotlib import animation

# Definición de ecuaciones de movimiento
def newton(t, Y, q, m, B):
    """Computes the derivative of the state vector y according to the equation of motion:
    Y is the state vector (x, y, z, u, v, w) === (position, velocity).
    returns dY/dt.
    """
    x, y, z = Y[0], Y[1], Y[2]
    u, v, w = Y[3], Y[4], Y[5]

    alpha = q / m * B
    return np.array([u, v, w, 0, alpha * w, -alpha * v])

# Inicializamos el método de integración
r = ode(newton).set_integrator('dopri5')

# Condiciones iniciales
t0 = 0
x0 = np.array([0, 0, 0])
v0 = np.array([1, 1, 1])
initial_conditions = np.concatenate((x0, v0))

# Resolving the equations
r.set_initial_value(initial_conditions, t0).set_f_params(1.0, 1.0, 1.0)

positions = []
t1 = 50
dt = 0.05
while r.successful() and r.t < t1:
    r.integrate(r.t+dt)
    positions.append(r.y[:3])       # Guardo solo posiciones, no velocidad

positions = np.array(positions)     # Paso a array


# Procedo a graficar las trayectorias y hacer la animación
# La idea es que grafico la primera y voy actualizando las posiciones para
# crear los "cuadros" de la animación.

fig = plt.figure(1,figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2])
ax.set_xlim(-1,50)
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')

# Animation 1
FRAMES = 50
fig = plt.figure(1,figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

def init():
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

# animation function.  This is called sequentially
def animate(i):
    current_index = int(positions.shape[0] / FRAMES * i)
    ax.cla()
    ax.plot3D(positions[:current_index, 0],
              positions[:current_index, 1],
              positions[:current_index, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
# call the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=FRAMES, interval=100)

# call our new function to display the animation
anim.save('magnetic_field.gif', writer='imagemagick')
