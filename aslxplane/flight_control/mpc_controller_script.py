from casadi import *
import numpy as np
import json

import sys, os

def shift(T, t0, x0, u, f):
    st = x0
    con = u[:,0]
    f_value = f(st, con)
    st = st + T * f_value
    x0 = np.array(st).reshape(-1)
    t0 = t0 + T
    u0 = np.concatenate([u[1:], u[-1:]])
    return t0, x0, u0

f = open('../data/dynamics_linear.json')
 
# returns JSON object as 
# a dictionary
params = json.load(f)["params"]
Wx_loaded = params["Wx"]
Wu_loaded = params["Wu"]
b_loaded = params["b"]
dt = (params["dt_sqrt"][0])**2
hc = params["heading_correction"]

DEFAULT_COST_CONFIG = {
    "heading_cost": 1e4,
    "roll_cost": 3e4,
    "position_cost": 1e0,
    "altitude_cost": 1e2,
    "par_cost": 498.863996,
    "perp_cost": 481.605499,
    "perp_quad_cost": 0.002698,
    "par_quad_cost": 1e-3,
}

x = MX.sym('x') 
y = MX.sym('y')
z = MX.sym('z') 
v = MX.sym('v')
vh = MX.sym('vh')
th = MX.sym('th')
phi = MX.sym('phi')
psi = MX.sym('psi')
th_p = MX.sym('th_p')
phi_p = MX.sym('phi_p')
psi_p = MX.sym('psi_p')

states = vertcat(x,y,z,v,vh,th,phi,psi,th_p,phi_p,psi_p)
states_est = vertcat(vh,th,phi,psi,th_p,phi_p,psi_p)
n_states = MX.size(states)[0]

print(n_states)

# Thrust, Alieron, etc...?
c1 = MX.sym('c1')
c2 = MX.sym('c2')
c3 = MX.sym('c3')
c4 = MX.sym('c4')

Wx = DM(Wx_loaded)
Wu = DM(Wu_loaded)
b = DM(b_loaded)

controls = vertcat(c1,c2,c3,c4)
n_controls = MX.size(controls)[0]

rhs = vertcat(v*np.cos(th),v*np.sin(th), Wx@states_est + Wu@controls + b)
f = Function('f',[states,controls],[rhs]) # nonlinear mapping function f(x,u)

Np = 100
Nc = 50

U = MX.sym('U',n_controls,Nc) # Decision variables (controls)
X = MX.sym('X',n_states,(Np+1))

# Parameters during sim
P = MX.sym('P',n_states + n_states)

# # compute solution symbolically

X = []
X.append(P[0:n_states])

st = P[0:n_states]
for k in range(Np):
    con = U[:,min(Nc-1,k)]
    f_value  = f(st,con)
    st =  st + (dt*f_value)
    X.append(st)

X = horzcat(*X)

ff=Function('ff',[U,P],[X])

obj = 0; 

Q = np.zeros((n_states,n_states))

for i in range(n_states):
    Q[i,i] = 1
R = np.zeros((n_controls,n_controls))
R[1,1] = 2 # weighing matrices (controls)

Q = DM(Q)
R = DM(R)

# compute objective
for k in range(Np):
    st = X[:,k]  - P[n_states:]
    con = U[:,min(Nc-1,k)]
    obj = obj+st.T@Q@st + con.T@R@con ; # calculate obj

OPT_variables = vertcat(U).reshape((Nc*n_controls, 1))

nlp_prob = {'f': obj, 'x': OPT_variables, 'p': P}

opts = {
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.acceptable_tol': 1e-8,
    'ipopt.acceptable_obj_change_tol': 1e-6,
    'ipopt.max_iter': 100
}

solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

# compute constraints
# for k = 1:N+1   % box constraints due to the map margins
#     g = [g ; X(1,k)];   %state x
#     g = [g ; X(2,k)];   %state x_dot
# end

t0 = 0
x0 = 1000*np.ones((n_states,))  # initial condition
xs = 1*np.ones((n_states,))  # reference posture
u0 = np.zeros((n_controls*Nc,))  # control inputs
sim_tim = 40  # maximum simulation time
xx = np.zeros((11, int(sim_tim / dt) + 1)) # Stores History
xx[:, 0] = x0
t = np.zeros(int(sim_tim / dt) + 1)




mpciter = 0
xx1 = []
u_cl = []

while np.linalg.norm(x0 - xs) > 1e-2 and mpciter < int(sim_tim / dt):
    print(f"The Iter Number:{mpciter}")
    args = {'p': vertcat(x0, xs), 'x0': u0.reshape(-1, 1)}
    sol = solver(**args)
    u = np.array(sol['x']).reshape(n_controls,Nc)
    ff_value = ff(u, vertcat(x0, xs))
    xx1.append(np.array(ff_value))
    u_cl.append(u[0])
    t[mpciter] = t0
    t0, x0, u0 = shift(dt, t0, x0, u, f)
    xx[:, mpciter+1] = x0
    mpciter += 1

    print(f"The Error is:{np.linalg.norm(x0 - xs)}")

Tvec = np.linspace(0, sim_tim, len(xx[0]))











