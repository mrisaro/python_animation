import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
from matplotlib import animation

# Function definitions
def newton(t, Y, q, m, B):
    """Computes the derivative of the state vector y according to the equation of motion:
    Y is the state vector (x, y, z, u, v, w) === (position, velocity).
    returns dY/dt.
    """
    x, y, z = Y[0], Y[1], Y[2]
    u, v, w = Y[3], Y[4], Y[5]

    alpha = q / m * B
    return np.array([u, v, w, alpha * v, -alpha * u, 0])

# Method initialization
r = ode(newton).set_integrator('dopri5')

# Initial conditions
t0 = 0
x0 = np.array([0, 0, 0])
v0 = np.array([0, 1, 0])
initial_conditions = np.concatenate((x0, v0))

# Resolving the equations
r.set_initial_value(initial_conditions, t0).set_f_params(1.0, 1.0, 1.0)

positions = []
t1 = 3.15
dt = 0.01
while r.successful() and r.t < t1:
    r.integrate(r.t+dt)
    positions.append(r.y[:3]) # keeping only position, not velocity

positions = np.array(positions)
# Resolving other Mass
# Initial conditions
t0 = 0
x0 = np.array([0, 0, 0])
v0 = np.array([0, 1/np.sqrt(2), 0])
initial_conditions = np.concatenate((x0, v0))
r.set_initial_value(initial_conditions, t0).set_f_params(1.0, 2.0, 1.0)

positions2 = []
t1 = 3.15
dt = 0.01
while r.successful() and r.t < t1:
    r.integrate(r.t+dt)
    positions2.append(r.y[:3]) # keeping only position, not velocity

positions2 = np.array(positions2)

# Plot
fig = plt.figure(1,figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(positions[:, 0], positions[:, 1])
ax.plot([1,3],[0,0],'--',linewidth=3,color='C3')
ax.set_ylim(-1.1,1.1)
fig.tight_layout()
#plt.show(block=false)

# Animation 1
FRAMES = 50
fig = plt.figure(2,figsize=(8,6))
ax = fig.add_subplot(111)

def init():
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# animation function.  This is called sequentially
def animate(i):
    current_index = int(positions.shape[0] / FRAMES * i)
    ax.cla()
    ax.plot(positions[:current_index, 0],
              positions[:current_index, 1])
    ax.plot(positions2[:current_index, 0],positions2[:current_index, 1])
    ax.plot([1,3],[0,0],'--',linewidth=3,color='C3')
    ax.set_ylim(-1.5,1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
# call the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=FRAMES, interval=100)

# call our new function to display the animation
anim.save('magnetic_field_2d.gif', writer='imagemagick')
