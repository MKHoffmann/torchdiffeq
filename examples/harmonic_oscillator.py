# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:23:18 2023

@author: Matthias K. Hoffmann
"""

"""
Script for the integration of the quaternion Frenet-Serret formulas, with comparison of the integrators with and without projection.
"""

import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

def ode(t,x):
    return torch.tensor([[0., 1.],[-1., 0.]])@x
    

def get_projection(x0):
    l = torch.norm(x0)
    def norm_projection(x):
        # Conserve the norm of the initial condition
        return x/torch.norm(x)*l
    return norm_projection
#%%
x0 = torch.tensor([[1.,0.]]).T
t = torch.linspace(0,5,10)


fixed_solver = 'rk4'
adaptive_solver = 'adaptive_heun'

projection = get_projection(x0)

# Ground truth solution using dopri8
x_gt = odeint(ode, x0, t, method='dopri8')
x = odeint(ode, x0, t, method=fixed_solver)
x_proj = odeint(ode, x0, t, method=fixed_solver, options={'projection_fcn':projection})
x_adapt = odeint(ode, x0, t, method=adaptive_solver, options={'min_step':0.05})
x_adapt_proj = odeint(ode, x0, t, method=adaptive_solver, options={'projection_fcn':projection, "min_step":0.05})

# Print the distance in the euclidean distance between the ground truth and the projection solution for x.
print(f"Ground truth vs. base: {torch.norm(x_gt[-1,:]-x[-1,:])}")
print(f"Ground truth vs. projection: {torch.norm(x_gt[-1,:]-x_proj[-1,:])}")
print(f"Ground truth vs. adaptive: {torch.norm(x_gt[-1,:]-x_adapt[-1,:])}")
print(f"Ground truth vs. adaptive projection: {torch.norm(x_gt[-1,:]-x_adapt_proj[-1,:])}")

# Plot the 3D curves consisting of 4, 5, and 6.
fig = plt.figure()
plt.plot(x_gt[:,0],x_gt[:,1], label="Ground Truth")
plt.plot(x[:,0],x[:,1], label="Base")
plt.plot(x_proj[:,0],x_proj[:,1], label="Projection")
# plt.plot(x_adapt[:,0],x_adapt[:,1], label="Adaptive")
# plt.plot(x_adapt_proj[:,0],x_adapt_proj[:,1], label="Adaptive Projection")
plt.legend()
plt.axis('equal')
plt.show()
