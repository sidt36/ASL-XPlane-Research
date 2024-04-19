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

def Return_State_Transition_Function(states,states_est,controls,Model_Params):
    
    v = states[4]
    th = states[6]
    

    Wx = DM(Model_Params["Wx"])
    Wu = DM(Model_Params["Wu"])
    b = DM(Model_Params["b"])
    hc = DM(Model_Params["hc"])

    rhs = vertcat(v*np.cos(th + 0*hc),v*np.sin(th + 0*hc), Wx@states_est + Wu@controls + b)

    f = Function('f',[states,controls],[rhs])

    return f

def Return_Objective(Q,R,Np,Nc,X,U,P,n_states):

    obj = 0
    Q = DM(Q)
    R = DM(R)
    for k in range(Np):
        st = X[:,k]  - P[n_states:]
        con = U[:,min(Nc-1,k)]
        obj = obj+st.T@Q@st + con.T@R@con # calculate obj
    
    return obj

def Return_Optimization_Setup(obj,U,P,Nc,n_controls):
    
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

    return solver







