from ConfinedCompression import *

from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import time as time
import matplotlib.pyplot as plt
from ngsolve.krylovspace import GMRes
import matplotlib.pyplot as plt
import multiprocessing as mp

# Material parameters
G = 3e4 # [Pa] Shear modulus
nu = 0.2 # Poisson's ratio
viscosity = 1e-3 # [N s] Fluid viscosity
alpha = 0.9 # Biot coefficient
n = 0.75 # Porosity
k = 1e-16 # [m^2] Intrinsic Permeability
chi = 1e-10 # [Pa^-1] Fluid Compressibility
rho = 1000 # [kg/m^3] Fluid density
g = 9.81 # [m/s^2] Gravitational acceleration

# simulation parameters
dt = 0.01 # [s] Starting Time step size
dt_max = 10 # [s] Maximum time step size for dynamic time stepping
dt_growth = 1.05 # Growth factor for dynamic time stepping
R = 3.6e-3 # [m] Radius of the cylindrical sample
H = 1e-3 # [m] Height of the cylindrical sample
u_max = 1e-4 # [m] Maximum displacement of the compression
compression_time = 1 # [s] Time duration for which the compression is applied
t_end = 300 # [s] Total simulation time
h = 0.5 # [Non Dimensional] Maximum mesh element size
order = 2 # Polynomial order for the finite element space

# Timing code
t0 = time.time()
time_vals, F_solid, F_fluid = ConfinedCompression(G, nu, viscosity, alpha, n, k, chi, rho, dt, dt_max, dt_growth, R, H, u_max , compression_time, t_end, h, order)
t1 = time.time()
print("Simulation time: ", t1 - t0, " seconds")
plt.plot(time_vals, F_solid,'--', color='green',lw=2,  label='Solid Force')
plt.plot(time_vals, F_fluid,'--', color='blue',lw=2,  label='Fluid Force')
plt.plot(np.array(time_vals), np.array(F_solid) + np.array(F_fluid), label='Total Force', linestyle='-', color='black', lw=3)
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.title('Confined Compression: Force vs Time')
plt.legend()
plt.grid()
plt.show()
