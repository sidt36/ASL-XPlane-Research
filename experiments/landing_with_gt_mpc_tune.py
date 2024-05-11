from __future__ import annotations

from pathlib import Path
import random
from argparse import ArgumentParser
from ray.tune.schedulers import AsyncHyperBandScheduler

import json
from ray import tune

import numpy as np
from tqdm import tqdm
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.air import RunConfig
import decimal
import time
from franges import  frange
from ray.tune.schedulers import ASHAScheduler

from aslxplane import xpc
from aslxplane.flight_control import utils
from aslxplane.flight_control.utils import RobustXPlaneConnect, FlightState
from aslxplane.flight_control.utils import deg2rad, rad2deg, reset_flight
from aslxplane.utils.video_recording import VideoRecorder
from aslxplane.flight_control.lqr_controller import LQRFlightController, DEFAULT_COST_CONFIG
from aslxplane.flight_control.mpc_controller import MPCFlightController
from aslxplane.simulation.weather import set_the_weather


def cost_fn(x_hist, target, approach_ang):
    x_hist, target = np.array(x_hist), np.array(target)[:2]
    dx = target[:2] - x_hist[:, :2]
    min_approach_idx = np.argmin(np.linalg.norm(dx, axis=-1))
    approach_alt = x_hist[min_approach_idx, 2]
    if approach_alt > 25.0:
        return 1e9
    v_norm = np.array([np.cos(approach_ang), np.sin(approach_ang)])
    v_par = np.sum(dx * v_norm[None, :], -1)[:, None] * v_norm[None, :]
    v_perp = dx - v_par
    return float(np.linalg.norm(v_perp, axis=-1)[min_approach_idx])


def run_trial(
    cost_config: dict[str, float],
    sim_speed: float = 1.0,
    record_video: bool = False,
    view: xpc.ViewType = None,
    abort_at: float = 100,  # seconds
    offsets_to_test: list[tuple[int]] = [(0, 0)],
    display: bool = False,
    data_prefix: str | Path = ".",
    randomize_weather: bool = True,
):
    trial_id = random.randint(0, int(1e6) - 1)
    reset_flight(RobustXPlaneConnect(), on_crash_only=False)
    controller = MPCFlightController(config={"sim_speed": sim_speed}, view=view)
    hist_list, cost_list = [], []
    recorder = None
    for offset in tqdm(offsets_to_test):
        if randomize_weather:
            weather_desc = set_the_weather()  # no specification means random
            if record_video:
                weather_path = Path(data_prefix) / Path(f"recording_weather_{trial_id:06d}.json")
                weather_path.write_text(
                    json.dumps({k: np.array(v).tolist() for (k, v) in weather_desc.items()})
                )
        controller.cost_config.update(cost_config)
        controller.config.update({"x0_offset": offset[0], "y0_offset": offset[1]})
        if record_video:
            recording_path = Path(data_prefix) / f"recording_{trial_id:06d}"
            recorder = VideoRecorder(recording_path, controller.flight_state, 0, display=display)
        crashed = controller.loop(how_long=120)
        controller.controller = "pid"
        controller.done = False
        if crashed:
            if recorder is not None:
                recorder.close()
            controller.close()
            return dict(objective=1e9)
        hist = {
            "x": [x.tolist() for x in controller.x_hist],
            "u": [z.tolist() for z in controller.u_hist],
            "t": controller.t_hist,
        }
        cost = cost_fn(hist["x"], target=controller.target, approach_ang=controller.approach_ang)
        cost_list.append(cost)
        print(f"cost = {cost:.4e}")
        hist_list.append(dict(offset=np.array(offset).tolist(), hist=hist))
        if cost > 1e8:
            break
    objective = np.max(cost_list)
    if recorder is not None:
        recorder.close()
    controller.close()
    return dict(
        objective=objective, cost_list=json.dumps(cost_list), hist_list=json.dumps(hist_list)
    )



def run_trial_cost(
    cost_config: dict[str, float],
    sim_speed: float = 1.0,
    record_video: bool = False,
    view: xpc.ViewType = None,
    abort_at: float = 100,  # seconds
    offsets_to_test: list[tuple[int]] = [(0, 0)],
    display: bool = False,
    data_prefix: str | Path = ".",
    randomize_weather: bool = True,
):
    trial_id = random.randint(0, int(1e6) - 1)
    reset_flight(RobustXPlaneConnect(), on_crash_only=False)
    time.sleep(2.0)
    controller = MPCFlightController(config={"sim_speed": sim_speed},cost_config = cost_config, view=view)
    hist_list, cost_list = [], []
    recorder = None
    for offset in tqdm(offsets_to_test):
        if randomize_weather:
            weather_desc = set_the_weather()  # no specification means random
            if record_video:
                weather_path = Path(data_prefix) / Path(f"recording_weather_{trial_id:06d}.json")
                weather_path.write_text(
                    json.dumps({k: np.array(v).tolist() for (k, v) in weather_desc.items()})
                )
        controller.cost_config.update(cost_config)
        controller.config.update({"x0_offset": offset[0], "y0_offset": offset[1]})
        if record_video:
            recording_path = Path(data_prefix) / f"recording_{trial_id:06d}"
            recorder = VideoRecorder(recording_path, controller.flight_state, 0, display=display)
        crashed = controller.loop(how_long=120)
        controller.controller = "pid"
        controller.done = False
        if crashed:
            if recorder is not None:
                recorder.close()
            controller.close()
            return dict(objective=1e9)
        hist = {
            "x": [x.tolist() for x in controller.x_hist],
            "u": [z.tolist() for z in controller.u_hist],
            "t": controller.t_hist,
        }
        cost = cost_fn(hist["x"], target=controller.target, approach_ang=controller.approach_ang)
        cost_list.append(cost)
        print(f"cost = {cost:.4e}")
        hist_list.append(dict(offset=np.array(offset).tolist(), hist=hist))
        if cost > 1e8:
            break
    objective = np.max(cost_list)
    if recorder is not None:
        recorder.close()
    controller.close()
    return dict(
        objective=objective, cost_list=json.dumps(cost_list), hist_list=json.dumps(hist_list)
    )
####################################################################################################

    

DEFAULT_COST_CONFIG = {
    "heading_cost": 1e5,
    "roll_cost": 3e4,
    "position_cost": 1e0*8,
    "altitude_cost": 1.12e3,
    "par_cost": 498.863996,
    "perp_cost": 481.605499,
    "perp_quad_cost": 0.002698,
    "par_quad_cost": 1e-3,
    "r_1":5e2,
    "r_2":1.5e4,
    "r_3":5e5,
    "r_4":1e3
}

Search_Space = {
    "heading_cost": 1e5,
    "roll_cost": 3e4,
    "position_cost": tune.grid_search([1e-1,5e-1,1,2,3,5,6,8,25,50]),
    "altitude_cost": tune.grid_search(list(range(1000,5000,100))),
    "par_cost": 498.863996,
    "perp_cost": 481.605499,
    "perp_quad_cost": 0.002698,
    "par_quad_cost": 1e-3,
    "r_1":tune.grid_search(list(range(400,1500,100))),
    "r_2":tune.grid_search(list(range(1000,20000,2000))),
    "r_3":tune.grid_search(list(range(50000,1000000,10000))),
    "r_4":tune.grid_search(list(range(700,1500,100)))
}
class TimeBudgetStopper(tune.Stopper):
    def __init__(self, time_budget_s):
        self._start_time = time.time()
        self._time_budget_s = time_budget_s

    def __call__(self, trial_id, result):
        return time.time() - self._start_time > self._time_budget_s

    def stop_all(self):
        return time.time() - self._start_time > self._time_budget_s



def objective(config):

    return run_trial_cost(
            config,
            sim_speed=1.0,
            record_video=False,
            display=False,
            view=xpc.ViewType.FullscreenNoHud,
            # view=xpc.ViewType.Chase,
            # offsets_to_test=[offset],
            # data_prefix=Path("~/datasets/xplane_recording5").expanduser(),
        )


def main():

    tuner = tune.Tuner(objective, param_space=Search_Space,tune_config=tune.TuneConfig(max_concurrent_trials=1), run_config=RunConfig(
        stop={"time_total_s": 100},  # 100 seconds
    ))

    results = tuner.fit()

    # stopper = TimeBudgetStopper(time_budget_s=45) 
    # # scheduler = AsyncHyperBandScheduler(max_concurrent=1)
    # analysis = tune.run(
    #     objective,
    #     config=Search_Space,
    #     stop=stopper,
    #     num_samples=1,
    #     resources_per_trial={"cpu": 16, "gpu": 0},
    #     local_dir='./ray_results',
    #     log_to_file=True
    #     )


    print(results)


if __name__ == "__main__":
    main()
