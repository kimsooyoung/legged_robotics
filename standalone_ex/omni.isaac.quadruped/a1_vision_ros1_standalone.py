# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
Introduction:

In this demo, the quadruped is publishing data from a pair of stereovision cameras and imu data for the VINS fusion 
visual interial odometry algorithm. Users can use the keyboard mapping to control the motion of the quadruped while the
quadruped localize itself.
"""


from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.quadruped.robots import UnitreeVision
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.nucleus import get_assets_root_path
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
import sensor_msgs.msg as sensor_msgs
import rospy


class A1_stereo_vision(object):
    def __init__(self, physics_dt, render_dt) -> None:
        """
        [Summary]

        creates the simulation world with preset physics_dt and render_dt and creates a unitree a1 robot (with ros cameras) inside a custom
        environment, set up ros publishers for the isaac_a1/imu_data and isaac_a1/foot_force topic

        Argument:
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.
        
        """
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        prim = get_prim_at_path("/World/Warehouse")
        if not prim.IsValid():
            prim = define_prim("/World/Warehouse", "Xform")
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets server")
            asset_path = assets_root_path + "/Isaac/Samples/ROS/Scenario/visual_odometry_testing.usd"

            prim.GetReferences().AddReference(asset_path)

        self._a1 = self._world.scene.add(
            UnitreeVision(
                prim_path="/World/A1", name="A1", position=np.array([0, 0, 0.27]), physics_dt=physics_dt, model="A1"
            )
        )
        # Publish camera images every 3 frames
        simulation_app.update()
        self._a1.setCameraExeutionStep(3)
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

        self._footforce_pub = rospy.Publisher("isaac_a1/foot_force", Float32MultiArray, queue_size=10)
        self._imu_pub = rospy.Publisher("isaac_a1/imu_data", sensor_msgs.Imu, queue_size=21)

        self._step_count = 0
        self._publish_interval = 2

        self._foot_force = Float32MultiArray()

        self._imu_msg = sensor_msgs.Imu()
        self._imu_msg.header.frame_id = "base_link"

    def setup(self) -> None:
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

    def on_physics_step(self, step_size) -> None:
        """
        [Summary]

        Physics call back, switch robot mode and call robot advance function to compute and apply joint torque
        
        """
        if self._event_flag:
            self._a1._qp_controller.switch_mode()
            self._event_flag = False

        self._a1.advance(step_size, self._base_command)
        og.Controller.evaluate_sync(self._clock_graph)

        self._step_count += 1

        if self._step_count % self._publish_interval == 0:
            ros_time = rospy.get_rostime()
            self.update_footforce_data()
            self._footforce_pub.publish(self._foot_force)
            self.update_imu_data()
            self._imu_msg.header.stamp = ros_time
            self._imu_pub.publish(self._imu_msg)
            self._step_count = 0

    def update_footforce_data(self) -> None:
        """
        [Summary]

        Update foot position and foot force data for ros publisher

        """
        self._foot_force.data = np.concatenate(
            (self._a1.foot_force, self._a1._qp_controller._ctrl_states._foot_pos_abs[:, 2])
        )

    def update_imu_data(self) -> None:
        """
        [Summary]

        Update imu data for ros publisher

        """
        self._imu_msg.orientation.x = self._a1._state.base_frame.quat[0]
        self._imu_msg.orientation.y = self._a1._state.base_frame.quat[1]
        self._imu_msg.orientation.z = self._a1._state.base_frame.quat[2]
        self._imu_msg.orientation.w = self._a1._state.base_frame.quat[3]

        self._imu_msg.linear_acceleration.x = self._a1._measurement.base_lin_acc[0]
        self._imu_msg.linear_acceleration.y = self._a1._measurement.base_lin_acc[1]
        self._imu_msg.linear_acceleration.z = self._a1._measurement.base_lin_acc[2]

        self._imu_msg.angular_velocity.x = self._a1._measurement.base_ang_vel[0]
        self._imu_msg.angular_velocity.y = self._a1._measurement.base_ang_vel[1]
        self._imu_msg.angular_velocity.z = self._a1._measurement.base_ang_vel[2]

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
        
        """  # reset event
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
    rospy.init_node("isaac_a1", anonymous=False, disable_signals=True, log_level=rospy.ERROR)
    rospy.set_param("use_sim_time", True)
    physics_downtime = 1 / 400.0
    runner = A1_stereo_vision(physics_dt=physics_downtime, render_dt=8 * physics_downtime)
    simulation_app.update()
    runner.setup()

    # an extra reset is needed to register
    runner._world.reset()
    runner._world.reset()
    runner.run()
    rospy.signal_shutdown("a1 vision complete")
    simulation_app.close()


if __name__ == "__main__":
    main()
