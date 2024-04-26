from __future__ import annotations

from pathlib import Path
import random
from argparse import ArgumentParser
import json

import numpy as np
from tqdm import tqdm
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.air import RunConfig


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

    

COST_CONFIG = {
    "heading_cost": 1e4,
    "roll_cost": 3e4,
    "position_cost": 1e0,
    "altitude_cost": 1e2,
    "par_cost": 498.863996,
    "perp_cost": 481.605499,
    "perp_quad_cost": 0.002698,
    "par_quad_cost": 1e-3,
}

Search_Space = {
    "heading_cost": tune.grid_search(range(1, 100000,100)),
    "roll_cost": tune.grid_search(range(1, 100000,100)),
    "position_cost": tune.grid_search([1e-2,1e-1,1e0,1e1,1e2]),
    "altitude_cost": tune.grid_search([1e-1,1e0,1e1,1e2,1e3]),
     "par_cost": 498.863996,
    "perp_cost": 481.605499,
    "perp_quad_cost": 0.002698,
    "par_quad_cost": 1e-3,
}

def objective(config):

    return run_trial_cost(
            config,
            sim_speed=1.0,
            record_video=True,
            display=False,
            view=xpc.ViewType.FullscreenNoHud,
            # view=xpc.ViewType.Chase,
            # offsets_to_test=[offset],
            # data_prefix=Path("~/datasets/xplane_recording5").expanduser(),
        )


def main():

    tuner = tune.Tuner(objective, param_space=COST_CONFIG)

    results = tuner.fit()

    print(results)


if __name__ == "__main__":
    main()
