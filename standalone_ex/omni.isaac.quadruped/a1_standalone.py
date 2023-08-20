# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.quadruped.robots import Unitree
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import Gf, UsdGeom

import omni.appwindow  # Contains handle to keyboard
import numpy as np
import carb
import argparse
import json


class A1_runner(object):
    def __init__(self, physics_dt, render_dt, way_points=None) -> None:
        """
        Summary

        creates the simulation world with preset physics_dt and render_dt and creates a unitree a1 robot inside the warehouse

        Argument:
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.
        way_points {List[List[float]]} -- x coordinate, y coordinate, heading (in rad) 
        
        """
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # spawn warehouse scene
        prim = get_prim_at_path("/World/Warehouse")
        if not prim.IsValid():
            prim = define_prim("/World/Warehouse", "Xform")
            asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
            prim.GetReferences().AddReference(asset_path)

        self._a1 = self._world.scene.add(
            Unitree(
                prim_path="/World/A1",
                name="A1",
                position=np.array([0, 0, 0.40]),
                physics_dt=physics_dt,
                model="A1",
                way_points=way_points,
            )
        )

        self._world.reset()
        self._enter_toggled = 0
        self._base_command = [0.0, 0.0, 0.0, 0]
        self._event_flag = False
        # bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [1.8, 0.0, 0.0],
            "UP": [1.8, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-1.8, 0.0, 0.0],
            "DOWN": [-1.8, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -1.8, 0.0],
            "RIGHT": [0.0, -1.8, 0.0],
            # right command
            "NUMPAD_4": [0.0, 1.8, 0.0],
            "LEFT": [0.0, 1.8, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 1.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -1.0],
            "M": [0.0, 0.0, -1.0],
        }

    def setup(self, way_points=None) -> None:
        """
        [Summary]

        Set unitree robot's default stance, set up keyboard listener and add physics callback
        
        """

        self._a1.set_state(self._a1._default_a1_state)
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._world.add_physics_callback("a1_advance", callback_fn=self.on_physics_step)

        if way_points is None:
            self._path_follow = False
        else:
            self._path_follow = True

    def on_physics_step(self, step_size) -> None:
        """
        [Summary]

        Physics call back, switch robot mode and call robot advance function to compute and apply joint torque
        
        """

        if self._event_flag:
            self._a1._qp_controller.switch_mode()
            self._event_flag = False

        self._a1.advance(step_size, self._base_command, self._path_follow)

    def run(self) -> None:
        """
        [Summary]

        Step simulation based on rendering downtime
        
        """
        # change to sim running
        while simulation_app.is_running():
            self._world.step(render=True)
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """
        [Summary]

        Keyboard subscriber callback to when kit is updated.
        
        """
        # reset event
        self._event_flag = False
        # when a key is pressed for released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command[0:3] += np.array(self._input_keyboard_mapping[event.input.name])
                self._event_flag = True

            # enter, toggle the last command
            if event.input.name == "ENTER" and self._enter_toggled is False:
                self._enter_toggled = True
                if self._base_command[3] == 0:
                    self._base_command[3] = 1
                else:
                    self._base_command[3] = 0
                self._event_flag = True

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command[0:3] -= np.array(self._input_keyboard_mapping[event.input.name])
                self._event_flag = True
            # enter, toggle the last command
            if event.input.name == "ENTER":
                self._enter_toggled = False
        # since no error, we are fine :)
        return True


parser = argparse.ArgumentParser(description="a1 quadruped demo")
parser.add_argument("-w", "--waypoint", type=str, metavar="", required=False, help="file path to the waypoints")
args, unknown = parser.parse_known_args()


def main():
    """
    [Summary]

    Parse arguments and instantiate A1 runner
    
    """
    physics_downtime = 1 / 400.0
    if args.waypoint:
        waypoint_pose = []
        try:
            print(str(args.waypoint))
            file = open(str(args.waypoint))
            waypoint_data = json.load(file)
            for waypoint in waypoint_data:
                waypoint_pose.append(np.array([waypoint["x"], waypoint["y"], waypoint["rad"]]))
            # print(str(waypoint_pose))

        except FileNotFoundError:
            print("error file not found, ending")
            simulation_app.close()
            return

        runner = A1_runner(physics_dt=physics_downtime, render_dt=16 * physics_downtime, way_points=waypoint_pose)
        simulation_app.update()
        runner.setup(way_points=waypoint)
    else:
        runner = A1_runner(physics_dt=physics_downtime, render_dt=16 * physics_downtime, way_points=None)
        simulation_app.update()
        runner.setup(None)

    # an extra reset is needed to register
    runner._world.reset()
    runner._world.reset()
    runner.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
