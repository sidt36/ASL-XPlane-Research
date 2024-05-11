
from __future__ import annotations

import os
import time
from pathlib import Path
import json
import math
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Optional
from multiprocessing import Lock
import matplotlib.pyplot as plt
import sys, os
from casadi import *


os.environ["JAX_PLATFORM_NAME"] = "CPU"

import numpy as np

from .. import xpc
from . import dynamics, utils
from .lqr_utils import design_LQR_controller
from .utils import RobustXPlaneConnect, deg2rad, rad2deg, FlightState, reset_flight
from .utils import FlightStateWithVision
from .utils import LATLON_DEG_TO_METERS
from .mpc_utils import Return_Params, Return_State_MX, Return_Controls_MX, Return_State_Transition_Function
from .mpc_utils import Return_Objective, Return_Optimization_Setup, advanceStateNumpy

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_PLATFORM_NAME"] = "CPU"
os.environ["JAX_ENABLE_X64"] = "True"

from jaxfi import jaxm

jaxm.set_default_dtype(np.float64)

####################################################################################################

DEFAULT_CONFIG = {
    "sim_speed": 1.0,
    "x0_offset": 0.0,
    "y0_offset": 0.0,
}

# magic numbers for Cessna Skyhawk
DEFAULT_COST_CONFIG = {
    "heading_cost": 1e5,
    "roll_cost": 3e4,
    "position_cost": 1e0*8,
    "altitude_cost": 1.1e3,
    "par_cost": 498.863996,
    "perp_cost": 481.605499,
    "perp_quad_cost": 0.002698,
    "par_quad_cost": 1e-3,
    "r_1":10e1*5e3,
    "r_2":3e0*5e3,
    "r_3":1e2*5e3,
    "r_4":2e-1*5e3

}
# weird angle misalignment vs runway, perhaps due to in-game magnetic compass distortion
ANGLE_CORRECTION = -0.1


class MPCFlightController:
    def __init__(
        self,
        config: dict[str, Any] = DEFAULT_CONFIG,
        cost_config: dict[str, float] = DEFAULT_COST_CONFIG,
        view: Optional[xpc.ViewType] = None,
        angle_correction: float = ANGLE_CORRECTION,
        open_loop = False
    ):
        """LQR Flight Controller - mostly tested for landing or heading line navigation.

        Args:
            config (dict[str, Any], optional): Defaults to DEFAULT_CONFIG.
            cost_config (dict[str, float], optional): LQR cost config. Defaults
                                                      to DEFAULT_COST_CONFIG.
            view (Optional[xpc.ViewType], optional): Which xpc.ViewType to use
                                                     on the plane. Defaults to None.
            angle_correction (float, optional): Landing strip angle relative to
                                                flight reset position. Defaults
                                                to ANGLE_CORRECTION.
        """
        self.open_loop = open_loop
        self.config, self.cost_config = deepcopy(DEFAULT_CONFIG), deepcopy(DEFAULT_COST_CONFIG)
        self.config.update(deepcopy(config))
        self.cost_config.update(deepcopy(cost_config))

        self.xp = RobustXPlaneConnect()
        self.int_state = np.zeros(6)
        self.vis_flight_state, self.use_vision = None, self.config.get("use_vision", None)
        self.flight_state = FlightState()
        if self.use_vision:
            self.vis_flight_state = FlightStateWithVision(
                **self.config.get("vision_config", dict())
            )
        while not np.all(np.isfinite(self.flight_state.last_sim_time_and_state[1])):
            time.sleep(1e-1)
        if self.use_vision:
            # wait for the vision state to initialize e.g., a deep model
            while not np.all(np.isfinite(self.vis_flight_state.last_sim_time_and_state[1])):
                time.sleep(1e-1)

        # initialize reference control as a return to initial state
        time.sleep(0.3)
        self.state0, self.posi0 = self.get_curr_state(), self.xp.getPOSI()
        self.target = self.get_curr_state()
        self.view = view if view is not None else xpc.ViewType.Chase

        # touchdown control parameters
        self.v_ref = 50.0
        self.params = dict()
        self.params["pos_ref"] = np.array([0.0, 0.0, 300.0])
        self.params["ang_ref"] = np.array([deg2rad(0.0), 0.0, deg2rad(self.posi0[5])])

        self.set_brake(0)
        self.t_start = time.time()
        self.u_hist, self.x_hist, self.x_vis_hist, self.t_hist = [], [], [], []
        self.controller = "mpc"
        self._read_dynamics()

        # runway specific angle correction
        self.approach_ang = deg2rad(self.posi0[5]) + angle_correction
        self.lock = Lock()
        self.ts, self.X, self.U, self.Ls = None, None, None, None
        self.done = False
        self.reset()
        time.sleep(2.0)

        self.data = dict()
    def plot_paths(self):
        plt.plot(np.array(self.x_hist)[0,0],np.array(self.x_hist)[0,1],'o')
        plt.plot(np.array(self.x_hist)[:,0],np.array(self.x_hist)[:,1])
        plt.plot(np.array(self.x_hist)[-1,0],np.array(self.x_hist)[-1,1],'x')
        plt.plot(np.array(self.target)[0],np.array(self.target)[1],'*')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Path of Aircraft, Open Loop = {self.open_loop}')
        plt.savefig(str(time.time())+ ".png") 
    def loop(self, how_long: float = 30) -> None:
        """Apply control in a loop.

        Args:
            how_long (float, optional): How "real-time" long to run the loop at.
                                        Defaults to math.inf.
        """

        t_loop_start = time.time()
        self.reset()
        time.sleep(1)

        self.data = dict()
        self.it = 0
        self.u_hist, self.x_hist, self.t_hist = [], [], []
        self.t_start = time.time()
        self.u0 = None
        self.solver = None
        # T = 100.0
        # T = 110.0 / self.config["sim_speed"]
        dt_small = 1.0 / 50.0
        t_prev = 0.0
        while not self.done and time.time() - t_loop_start < how_long:
            t_prev = time.time()
            self.apply_control()
            sleep_for = max(0, dt_small - (time.time() - t_prev))
            time.sleep(sleep_for)
            self.it += 1
            is_crashed = self.xp.getDREF("sim/flightmodel2/misc/has_crashed")[0] > 0.0
            if is_crashed:
                # self.plot_paths()
                reset_flight(self.xp)
                time.sleep(2.0)
                return True
        if self.done:
            # self.plot_paths()
            self.reset()
            time.sleep(2.0)
        return False

    @staticmethod
    def _build_control(pitch=0, roll=0, yaw=0, throttle=0, gear=0, flaps=0):
        return [min(max(x, -1), 1) for x in [pitch, roll, yaw]] + [
            min(max(x, 0), 1) for x in [throttle, gear, flaps]
        ]

    def set_brake(self, brake: float = 1) -> None:
        self.xp.sendDREF(utils.BRAKE, brake)

    def reset(self):
        """Reset the simulation to the state about 5km in the air behind the runway."""
        self.xp.sendDREF(utils.SIM_SPEED, self.config["sim_speed"])
        self.xp.sendVIEW(self.view)
        for _ in range(1):
            # arrest speed
            self.xp.sendPOSI(self.posi0)
            self.xp.sendDREFs(list(utils.SPEEDS.values()), [0 for _ in utils.SPEEDS.values()])
            # arrest rotation
            self.xp.sendDREFs(
                list(utils.ROTATION_SPEEDS.values()), [0 for _ in utils.ROTATION_SPEEDS.values()]
            )
            self.xp.sendPOSI(self.posi0)
            self.xp.sendCTRL(self._build_control())
            self.set_brake()

            posi = list(copy(self.posi0))
            posi[2] = 300
            dist = 6e3
            # posi[0] += dist / LONLAT_DEG_TO_METERS * -math.cos(deg2rad(posi[5])) + 3e3 / DEG_TO_METERS
            # posi[1] += dist / LONLAT_DEG_TO_METERS * -math.sin(deg2rad(posi[5])) + 3e3 / DEG_TO_METERS
            posi[0] += (
                dist / LATLON_DEG_TO_METERS * -math.cos(deg2rad(posi[5]))
                + self.config["x0_offset"] / LATLON_DEG_TO_METERS
            )
            posi[1] += (
                dist / LATLON_DEG_TO_METERS * -math.sin(deg2rad(posi[5]))
                + self.config["y0_offset"] / LATLON_DEG_TO_METERS
            )


            posi[3] = 0
            posi[4] = 0
            posi[5] = 117.86


            # set the plane at the new reset position, match simulation speed to heading
            self.xp.sendPOSI(posi)
            v = 60.0
            vx, vz = v * math.sin(deg2rad(self.posi0[5])), v * -math.cos(deg2rad(self.posi0[5]))
            self.xp.sendDREFs([utils.SPEEDS["local_vx"], utils.SPEEDS["local_vz"]], [vx, vz])
            time.sleep(1)
        self.data = dict()

    def get_time_state(self):
        if self.use_vision:
            return tuple(self.vis_flight_state.last_sim_time_and_state)

        return tuple(self.flight_state.last_sim_time_and_state)

    def get_curr_time(self):
        return self.flight_state.last_sim_time_and_state[0]

    def get_curr_state(self, vision: bool = False) -> np.array:
        if vision and self.use_vision:
            state = self.vis_flight_state.last_sim_time_and_state[1]
        else:
            state = self.flight_state.last_sim_time_and_state[1]
        state = np.concatenate([np.array(state), self.int_state])
        return state
    
    def shift(self,u):
        """
        Shifts the control input to warm start the solver.

        Args:
            T: Timestep
            f: CasADi symbolic function.
            x0: Old Initial State
            u: Array of Control Inputs
            t0: new starting point.

        """
        u0 = np.concatenate([u[1:], u[-1:]])
        return  u0


    ################################################################################

    def _read_dynamics(self):
        dynamics_path = Path(__file__).absolute().parents[1] / "data" / "dynamics_linear.json"
        dynamics_state = json.loads(dynamics_path.read_text())
        fn = dynamics.int_f_fx_fu_fn
        self.params.update({k: jaxm.array(v) for (k, v) in dynamics_state["params"].items()})
        params = copy(self.params)
        self.f_fx_fu_fn = lambda x, u, *args: fn(x, u, params)

    def advance_state(self, dt):
        state = self.get_curr_state()
        pos_int = self.int_state[:3] + dt * (np.array(state[:3]) - self.params["pos_ref"])
        ang_int = self.int_state[3:6] + dt * (np.array(state[5:8]) - self.params["ang_ref"])
        int_state = np.concatenate([np.array(pos_int), np.array(ang_int)])
        self.int_state = 0.99**dt * int_state

        
    def _construct_mpc_problem(self, x0):
        #Setup 
        self.Model_Params= Return_Params()
        self.states, self.states_est, self.n_states = Return_State_MX()
        self.controls, self.n_controls = Return_Controls_MX()
        self.f = Return_State_Transition_Function(self.states,self.states_est,self.controls,self.Model_Params)

        self.Np = 20
        self.Nc = 20 

        self.U = MX.sym('U',self.n_controls,self.Nc) # Decision variables (controls)
        self.X = MX.sym('X',self.n_states,(self.Np+1))

        # Parameters during sim
        self.P = MX.sym('P',self.n_states + self.n_states)
        v_norm = np.array([math.cos(self.approach_ang), math.sin(self.approach_ang)])
        # Single Shooting in Time
        X = []
        X.append(self.P[0:self.n_states])
        st = self.P[0:self.n_states]
        dt = self.Model_Params["dt"]
        for k in range(self.Np):
            con = self.U[:,min(self.Nc-1,k)]
            f_value  = self.f(st,con)
            st =  st + (dt*f_value)
            X.append(st)

        self.X = horzcat(*X)
        self.ff=Function('ff',[self.U,self.P],[self.X])
        # Using LQR Weights

        x_ref = np.copy(x0)
        
        
        ##
        cc = self.cost_config
        q_diag = (
            np.array(
                [cc["position_cost"], cc["position_cost"], cc["altitude_cost"]]
                + [1e3, 1e0]
                + [1e0, cc["roll_cost"], cc["heading_cost"]]
                + [1e-3, 1e-3, 1e-3]
            )
            / 1e3
        )

        Q = np.diag(q_diag)



        R = np.diag(np.array((cc["r_1"], cc["r_2"], cc["r_3"],cc["r_4"]*0.89)))
        u_ref = np.array([0.0, 0.0, 0.0, 0.0])

        ##


        self.obj = Return_Objective(Q,R,self.Np,self.Nc,self.X,self.U,self.P,self.n_states,v_norm,cc)
        
        self.solver = Return_Optimization_Setup(self.obj,self.U,self.P,self.Nc,self.n_controls)


        return self.solver



    ################################################################################
    def compute_target_state(self,x0):
        target = self.target[:2] + 400 * np.array(  # 400 meters down the runway from starting point
            [math.cos(self.approach_ang), math.sin(self.approach_ang)]
        )
        x_ref = np.zeros((self.n_states,))
        v_norm = np.array([math.cos(self.approach_ang), math.sin(self.approach_ang)])
        dx = np.array(target[:2]) - np.array(x0[:2])
        v_par = np.sum(dx * v_norm) * v_norm
        v_perp = dx - v_par
        d_par = math.sqrt(max(5e2**2 - np.linalg.norm(v_perp) ** 2, 0)) / np.linalg.norm(v_par)
        x_ref[:2] = (
            x0[:2]
            + max(np.linalg.norm(v_perp), 1e2) * v_perp / np.linalg.norm(v_perp)
            + d_par * v_par
        )
        dist = np.linalg.norm(target[:2] - x0[:2])
        x_ref[2] = min(
            max(self.posi0[2], self.params["pos_ref"][2] * (dist / 5e3)), 300.0
        )  # altitude
        x_ref[3:5] = self.v_ref, 0.0  # velocities
        x_ref[5:8] = self.params["ang_ref"]

        return x_ref
    


    def apply_control(self):
        """Compute and apply the control action."""

        state = self.get_curr_state()
        vis_state = self.get_curr_state(vision=True)
        

        if self.controller == "pid":
            pitch, roll, heading = state[5:8]
            pitch_ref, roll_ref, heading_ref = deg2rad(5.0), 0.0, self.state0[7]
            u_pitch = -1.0 * (pitch - pitch_ref)
            u_roll = -1.0 * (roll - roll_ref)
            u_heading = -1.0 * (30.0 / state[3]) * (heading - heading_ref)
            throttle = 0.7
            u = np.array([u_pitch, u_roll, u_heading, throttle])
        elif self.controller == "lqr":
            Q, R, x_ref, u_ref = self._construct_lqr_problem(state)
            u0 = np.zeros(R.shape[-1])
            f, fx, fu = self.f_fx_fu_fn(state, u0)
            A, B, d = fx, fu, f - fx @ state - fu @ u0
            L, l = design_LQR_controller(A, B, d, Q, R, x_ref, u_ref, T=10)
            u = L @ state + l
        elif self.controller == "mpc":
            #TODO
            if(self.it%100==1):
                print("Iteration Number : ", self.it)
                print(f"Error is : {np.linalg.norm(self.x0 -  self.xs)}")
            if(self.it==0): 
                self.solver, self.lbx, self.ubx = self._construct_mpc_problem(state)
                x0 = state[0:11] # initial condition
                self.xs = self.compute_target_state(x0)  # reference posture
                self.u0 = np.zeros((self.n_controls*self.Nc,))  # control inputs
                args = {'p': vertcat(x0, self.xs), 'x0': self.u0.reshape(-1, 1), 'lbx': self.lbx, 'ubx': self.ubx}
                sol = self.solver(**args)
                u1 = np.array(sol['x']).reshape(self.n_controls,self.Nc)
                self.x0 = advanceStateNumpy(x0,u1[:,0])
                self.u0 = self.shift(u1)
            else:
                x0 = state[0:11] # initial condition
                if(self.open_loop):
                    x0 = self.x0
                self.xs = self.compute_target_state(x0)  # reference posture
                self.u0 = np.zeros((self.n_controls*self.Nc,))  # control inputs
                args = {'p': vertcat(x0, self.xs), 'x0': self.u0.reshape(-1, 1), 'lbx': self.lbx, 'ubx': self.ubx}
                sol = self.solver(**args)
                u1 = np.array(sol['x']).reshape(self.n_controls,self.Nc)
                self.x0 = advanceStateNumpy(x0,u1[:,0])
                self.u0 = self.shift(u1)
            u = u1[:,0]
        u_pitch, u_roll, u_heading, throttle = np.clip(u, [-1, -1, -1, 0], [1, 1, 1, 1])

        # landing stage, a poor man's finite state machine #########################################
        if state[2] < 5.0 or self.data.get("fixed_pitch", None) is not None:
            self.data.setdefault("fixed_pitch", u_pitch - 0.05)
            u_pitch, u_roll, u_heading, throttle = self.data["fixed_pitch"], 0.0, 0.0, 0.0
            self.set_brake(1)
        # landing stage, a poor man's finite state machine #########################################

        self.t_hist.append(self.get_curr_time())
        self.x_hist.append(copy(state))
        self.x_vis_hist.append(copy(vis_state))
        self.u_hist.append(copy(u))
        ctrl = self._build_control(pitch=u_pitch, roll=u_roll, yaw=u_heading, throttle=throttle)
        self.xp.sendCTRL(ctrl)

   
    def close(self):
        self.flight_state.close()
        self.done = True


####################################################################################################
