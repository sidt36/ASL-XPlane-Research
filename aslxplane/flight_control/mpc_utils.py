from casadi import *
import numpy as np
import json
from pathlib import Path

import sys, os


def Return_Params():

    dynamics_path = Path(__file__).absolute().parents[1] / "data" / "dynamics_linear.json"

    f = open(dynamics_path)
 
    # returns JSON object as 
    # a dictionary
    params = json.load(f)["params"]
    Wx_loaded = params["Wx"]
    Wu_loaded = params["Wu"]
    b_loaded = params["b"]
    dt = (params["dt_sqrt"][0])**2
    hc = params["heading_correction"]

    Model_Params = {}
    Model_Params["Wx"] = Wx_loaded
    Model_Params["Wu"] = Wu_loaded
    Model_Params["b"] = b_loaded
    Model_Params["dt"] = dt
    Model_Params["hc"] = hc

    return Model_Params

def Return_State_MX():

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

    return states,states_est, n_states

def Return_Controls_MX():
    c1 = MX.sym('c1')
    c2 = MX.sym('c2')
    c3 = MX.sym('c3')
    c4 = MX.sym('c4')

    controls = vertcat(c1,c2,c3,c4)
    n_controls = MX.size(controls)[0]

    return controls, n_controls

# From Robert's LQR Code
# def cost_fn(x0, target, v_norm):
#     """Compute a position cost as a scalar."""
#     dx = target[:2] - x0[:2]
#     v_par = jaxm.sum(dx * v_norm) * v_norm
#     v_perp = dx - v_par
#     v_perp_norm = jaxm.linalg.norm(v_perp)
#     v_perp_norm2 = jaxm.sum(v_perp**2)
#     v_par_norm = jaxm.linalg.norm(v_par)
#     cc = self.cost_config
#     Jv_perp = jaxm.where(
#         v_perp_norm > 1e3, v_perp_norm, cc["perp_quad_cost"] * v_perp_norm2
#     )
#     Jv_par = v_par_norm
#     return cc["perp_cost"] * Jv_perp + cc["par_cost"] * Jv_par

def Return_State_Transition_Function(states,states_est,controls,Model_Params):
    
    v = states[4]
    th = states[6]
    
    Wx = DM(Model_Params["Wx"])
    Wu = DM(Model_Params["Wu"])
    b = DM(Model_Params["b"])
    hc = DM(Model_Params["hc"])

    rhs = vertcat(v*cos(th + 0*hc),v*sin(th + 0*hc), Wx@states_est + Wu@controls + b)

    f = Function('f',[states,controls],[rhs])

    return f

def Return_Objective(Q,R,Np,Nc,X,U,P,n_states,v_norm,cc):

    obj = 0
    Q = MX(Q)
    R = DM(R)
    v_norm = DM(v_norm)


    x01 = MX.sym('x0', 2)
    target1 = MX.sym('target', 2)
    v_norm1 = MX.sym('v_norm', 2)

    x_ref = MX.sym('x_ref',n_states)
   
    def cost_fn(x0, target, v_norm):
        """Compute a position cost as a scalar."""
        dx = target[:2] - x0[:2]
        v_par = mtimes(mtimes(dx.T, v_norm), v_norm)
        v_perp = dx - v_par
        v_perp_norm = norm_2(v_perp)
        v_perp_norm2 = mtimes(v_perp.T, v_perp)
        v_par_norm = norm_2(v_par)
        Jv_perp = if_else(v_perp_norm > 1e3, v_perp_norm, cc["perp_quad_cost"] * v_perp_norm2)
        Jv_par = v_par_norm
        return cc["perp_cost"] * Jv_perp + cc["par_cost"] * Jv_par

    def cost_approx(x0, target, v_norm):
        """Develop a quadratic approximation of the cost function based on a scalar cost."""        
        g = gradient(cost_fn(x0, target, v_norm), x0)
        H = hessian(cost_fn(x0, target, v_norm), x0)[0]
        Q = H + 1e-3 * MX.eye(H.size1())
        ref = x0 - solve(Q, g)
        return Q, ref


    cost_approx_fn = Function('cost_approx_fn', [x01, target1, v_norm1], cost_approx(x01, target1, v_norm1))

    Qx, refx = cost_approx_fn(P[0:2], P[n_states:n_states+2], v_norm)

    # Qx_sub = MX.to_DM(Qx)


    Q[:2, :2] = Qx[:2, :2] / 1e3
    # Q = DM(Qx)
    x_ref = P[0:n_states]
    x_ref[:2] = refx[:2]
    for k in range(Np):
        st = X[:,k]  -x_ref
        con = U[:,min(Nc-1,k)]
        obj = obj+st.T@Q@st + con.T@R@con # calculate obj

    # obj = obj + cost_fn(P[0:n_states],P[n_states:],v_norm)
    
    return obj

def Return_Optimization_Setup(obj,U,P,Nc,n_controls):
    
    OPT_variables = vertcat(U).reshape((Nc*n_controls, 1))

    lbx = [-1]*(Nc*n_controls)
    ubx = [1]*(Nc*n_controls)

    nlp_prob = {'f': obj, 'x': OPT_variables, 'p': P}

    opts = {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6,
        'ipopt.max_iter': 100
    }

    solver = nlpsol('solver', 'ipopt', nlp_prob, opts)    

    return solver, lbx, ubx

def advanceStateNumpy(x0,u):
    Model_Params = Return_Params()
    v = x0[4]
    th = x0[6]
    f = np.zeros(x0.shape)
    f[0] = v*np.cos(th)
    f[1] = v*np.sin(th)
    f[2:] = Model_Params['Wx']@x0[4:] + Model_Params['Wu']@u + Model_Params['b']
    x_new = x0 + Model_Params["dt"]*f
    return x_new






