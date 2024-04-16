from casadi import *
import numpy as np
x = MX.sym("x")
print(jacobian(sin(x),x))

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

states = [x;y;z;v;vh;th;phi;psi;th_p;phi_p;psi_p]


c1 = MX.sym('c1')
c2 = MX.sym('c2')
c3 = MX.sym('c3')
c4 = MX.sym('c4')

controls = [c1;c2;c3;c4]

rhs = [v*np.cos(th);v*np.sin(th)]; 

f = function('f',{states,controls},{rhs}); # nonlinear mapping function f(x,u)
U = MX.sym('U',n_controls,N); % Decision variables (controls)

# Parameters
P = MX.sym('P',n_states + n_states+1)


X = MX.sym('X',n_states,(N+1))
# A Matrix that represents the states over the optimization problem.

# compute solution symbolically
X(:,0) = P(1:8); % initial state
for k in range(N):
    st = X(:,k);  con = U(:,k)
    f_value  = f(st,con)
    st_next  = st+ (T*f_value)
    X(:,k+1) = st_next






