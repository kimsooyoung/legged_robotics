# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
Introduction

This is a demo for the go1 robot's ros integration. In this example, the robot's foot position and contact forces are
being published to "/isaac_a1/output" topic, and these values can be plotted using plotjugler. 
"""


from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.quadruped.robots import Unitree
from omni.isaac.core.utils.extensions import enable_extension
import omni.appwindow  # Contains handle to keyboard
import numpy as np
import carb

import omni.graph.core as og

# enable ROS bridge extension
enable_extension("omni.isaac.ros_bridge")

simulation_app.update()

# check if rosmaster node is running
# this is to prevent this sample from waiting indefinetly if roscore is not running
# can be removed in regular usage
import rosgraph

if not rosgraph.is_master_online():
    carb.log_error("Please run roscore before executing this script")
    simulation_app.close()
    exit()

from std_msgs.msg import Float32MultiArray
import rospy


class Go1_runner(object):
    def __init__(self, physics_dt, render_dt) -> None:
        """
        [Summary]

        creates the simulation world with preset physics_dt and render_dt and creates a unitree go1 robot

        Argument:
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.     

        """
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        self._go1 = self._world.scene.add(
            Unitree(
                prim_path="/World/Go1", name="Go1", position=np.array([0, 0, 0.40]), physics_dt=physics_dt, model="Go1"
            )
        )
        self._world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
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

        # Creating an ondemand push graph with ROS Clock, everything in the ROS environment must synchronize with this clock
        try:
            keys = og.Controller.Keys
            (self._clock_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": "/ROS_Clock",
                    "evaluator_name": "push",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
                },
                {
                    keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("readSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("publishClock", "omni.isaac.ros_bridge.ROS1PublishClock"),
                    ],
                    keys.CONNECT: [
                        ("OnTick.outputs:tick", "publishClock.inputs:execIn"),
                        ("readSimTime.outputs:simulationTime", "publishClock.inputs:timeStamp"),
                    ],
                },
            )
        except Exception as e:
            print(e)
            simulation_app.close()
            exit()

        self._pub = rospy.Publisher("/isaac_a1/output", Float32MultiArray, queue_size=10)
        return

    def setup(self) -> None:
        """
        [Summary]

        Set unitree robot's default stance, set up keyboard listener and add physics callback
        
        """
        self._go1.set_state(self._go1._default_a1_state)
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._world.add_physics_callback("a1_advance", callback_fn=self.on_physics_step)

    def on_physics_step(self, step_size) -> None:
        """
        [Summary]

        Physics call back, switch robot mode and call robot advance function to compute and apply joint torque
        
        """
        if self._event_flag:
            self._go1._qp_controller.switch_mode()
            self._event_flag = False

        self._go1.advance(step_size, self._base_command)

        # Tick the ROS Clock
        og.Controller.evaluate_sync(self._clock_graph)

        self._pub.publish(Float32MultiArray(data=self.get_footforce_data()))

    def get_footforce_data(self) -> np.array:
        """
        [Summary]
        
        get foot force and position data
        """
        data = np.concatenate((self._go1.foot_force, self._go1._qp_controller._ctrl_states._foot_pos_abs[:, 2]))
        return data

    def run(self) -> None:
        """
        [Summary]

        Step simulation based on rendering downtime
        
        """
        # change to sim running
        while simulation_app.is_running():
            self._world.step(render=True)
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> None:
        """
        [Summary]
        
        Subscriber callback to when kit is updated.
        
        """
        # reset event
        self._event_flag = False
        # when a key is pressedor released  the command is adjusted w.r.t the key-mapping
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


def main() -> None:
    """
    [Summary]

    Instantiate ros node and start a1 runner
    
    """

    rospy.init_node("go1_standalone", anonymous=False, disable_signals=True, log_level=rospy.ERROR)
    rospy.set_param("use_sim_time", True)
    physics_downtime = 1 / 400.0
    runner = Go1_runner(physics_dt=physics_downtime, render_dt=16 * physics_downtime)
    simulation_app.update()
    runner.setup()

    # an extra reset is needed to register
    runner._world.reset()
    runner._world.reset()
    runner.run()
    rospy.signal_shutdown("go1 complete")
    simulation_app.close()


if __name__ == "__main__":
    main()
