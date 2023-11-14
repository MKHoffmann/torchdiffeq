"""
Script for the integration of the quaternion Frenet-Serret formulas, with comparison of the integrators with and without projection.
"""

import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

def get_quaternion_curvature_ode(u):
    # Get the ode for the quaternion Frenet-Serret formulas.
    def ode(t,x):
        q = x[:4,:]
        qm = torch.tensor([[0, -u[0], -u[1], -u[2]],
                           [u[0], 0, u[2], -u[1]],
                           [u[1], -u[2], 0, u[0]],
                           [u[2], u[1], -u[0], 0]])
        dq = qm@q
        dp = torch.tensor([[2*(q[1,:]*q[3,:]+q[0,:]*q[2,:])], [2*(q[2,:]*q[3,:]-q[0,:]*q[1,:])], [q[0,:]**2+q[3,:]**2-q[1,:]**2-q[2,:]**2]])
        
        return torch.cat([dq,dp],0)
    
    return ode


def quaternion_projection(x):
    # Project the quaternion onto the unit sphere.
    x[:4] = x[:4]/torch.norm(x[:4])
    return x

#%%
q0 = torch.tensor([[1.,0.,0.,0.]]).T
p0 = torch.zeros(3,1)
x0 = torch.cat([q0,p0],0)

t = torch.linspace(0,1,10)

u = torch.tensor([1.,2.,3.])

fixed_solver = 'euler'
adaptive_solver = 'adaptive_heun'

ode = get_quaternion_curvature_ode(u)

# Ground truth solution using dopri8
x_gt = odeint(ode, x0, t, method='dopri8')
x = odeint(ode, x0, t, method=fixed_solver)
x_proj = odeint(ode, x0, t, method=fixed_solver, options={'projection_fcn':quaternion_projection})
x_adapt = odeint(ode, x0, t, method=adaptive_solver, options={'min_step':0.05})
x_adapt_proj = odeint(ode, x0, t, method=adaptive_solver, options={'projection_fcn':quaternion_projection, "min_step":0.05})

# Print the distance in the euclidean distance between the ground truth and the projection solution for x[4:].
print(f"Ground truth vs. base: {torch.norm(x_gt[-1,4:]-x[-1,4:])}")
print(f"Ground truth vs. projection: {torch.norm(x_gt[-1,4:]-x_proj[-1,4:])}")
print(f"Ground truth vs. adaptive: {torch.norm(x_gt[-1,4:]-x_adapt[-1,4:])}")
print(f"Ground truth vs. adaptive projection: {torch.norm(x_gt[-1,4:]-x_adapt_proj[-1,4:])}")

# Plot the 3D curves consisting of 4, 5, and 6.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_gt[:,4],x_gt[:,5],x_gt[:,6], label="Ground Truth")
ax.plot(x[:,4],x[:,5],x[:,6], label="Base")
ax.plot(x_proj[:,4],x_proj[:,5],x_proj[:,6], label="Projection")
ax.plot(x_adapt[:,4],x_adapt[:,5],x_adapt[:,6], label="Adaptive")
ax.plot(x_adapt_proj[:,4],x_adapt_proj[:,5],x_adapt_proj[:,6], label="Adaptive Projection")
ax.legend()
plt.show()
