# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""
Introduction
We start a runner which publishes robot sensor data as ROS1 topics and listens to outside ROS1 topic "isaac_a1/joint_torque_cmd".
The runner set robot joint torques directly using the external ROS1 topic "isaac_a1/joint_torque_cmd".
The runner instantiate robot UnitreeDirect, which directly takes in joint torques and sends torques to lowlevel joint controllers
This is a very simple example to demonstrate how to treat Isaac Sim as a simulation component with in the ROS1 ecosystem
"""

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.quadruped.robots import UnitreeDirect
from omni.isaac.quadruped.utils.a1_classes import A1Measurement
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
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
# ros-python and ROS1 messages
import geometry_msgs.msg as geometry_msgs
import rospy
import sensor_msgs.msg as sensor_msgs


class A1_direct_runner(object):
    def __init__(self, physics_dt, render_dt) -> None:
        """
        [Summary]

        creates the simulation world with preset physics_dt and render_dt and creates a unitree a1 robot inside the warehouse

        Argument:
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.
        
        """

        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        prim = get_prim_at_path("/World/Warehouse")
        if not prim.IsValid():
            prim = define_prim("/World/Warehouse", "Xform")
            asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
            prim.GetReferences().AddReference(asset_path)

        self._a1 = self._world.scene.add(
            UnitreeDirect(
                prim_path="/World/A1", name="A1", position=np.array([0, 0, 0.40]), physics_dt=physics_dt, model="A1"
            )
        )

        self._world.reset()

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

        ##
        # ROS publishers
        ##
        # a) ground truth body pose
        self._pub_body_pose = rospy.Publisher("isaac_a1/gt_body_pose", geometry_msgs.PoseStamped, queue_size=21)
        self._msg_body_pose = geometry_msgs.PoseStamped()
        self._msg_body_pose.header.frame_id = "base_link"
        # b) joint angle and foot force
        self._pub_joint_state = rospy.Publisher("isaac_a1/joint_foot", sensor_msgs.JointState, queue_size=21)
        self._msg_joint_state = sensor_msgs.JointState()
        self._msg_joint_state.name = [
            "FL0",
            "FL1",
            "FL2",
            "FR0",
            "FR1",
            "FR2",
            "RL0",
            "RL1",
            "RL2",
            "RR0",
            "RR1",
            "RR2",
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        ]
        self._msg_joint_state.position = [0.0] * 16
        self._msg_joint_state.velocity = [0.0] * 16
        self._msg_joint_state.effort = [0.0] * 16
        # c) IMU measurements
        self._pub_imu_debug = rospy.Publisher("isaac_a1/imu_data", sensor_msgs.Imu, queue_size=21)
        self._msg_imu_debug = sensor_msgs.Imu()
        self._msg_imu_debug.header.frame_id = "base_link"

        # d) ground truth body pose with a fake covariance
        self._pub_body_pose_with_cov = rospy.Publisher(
            "isaac_a1/gt_body_pose_with_cov", geometry_msgs.PoseWithCovarianceStamped, queue_size=21
        )
        self._msg_body_pose_with_cov = geometry_msgs.PoseWithCovarianceStamped()
        self._msg_body_pose_with_cov.header.frame_id = "base_link"

        ##
        # ROS subscribers
        ##
        self._sub_joint_cmd = rospy.Subscriber(
            "isaac_a1/joint_torque_cmd", sensor_msgs.JointState, self.joint_command_callback
        )
        # buffer to store the robot command
        self._ros_command = np.zeros(12)

    def setup(self):
        """
        [Summary]

        add physics callback
        
        """
        self._app_window = omni.appwindow.get_default_app_window()
        self._world.add_physics_callback("robot_sim_step", callback_fn=self.robot_simulation_step)

        # start ROS publisher and subscribers

    def run(self):
        """
        [Summary]

        Step simulation based on rendering downtime
        
        """
        # change to sim running
        while simulation_app.is_running():
            self._world.step(render=True)
        return

    def publish_ros_data(self, measurement: A1Measurement):
        """
        [Summary]

        Publish body pose, joint state, imu data
        
        """
        # update all header timestamps
        ros_timestamp = rospy.get_rostime()
        self._msg_body_pose.header.stamp = ros_timestamp
        self._msg_joint_state.header.stamp = ros_timestamp
        self._msg_imu_debug.header.stamp = ros_timestamp
        self._msg_body_pose_with_cov.header.stamp = ros_timestamp

        # a) ground truth pose
        self._update_body_pose_msg(measurement)
        self._pub_body_pose.publish(self._msg_body_pose)
        # b) joint state and contact force
        self._update_msg_joint_state(measurement)
        self._pub_joint_state.publish(self._msg_joint_state)
        # c) IMU
        self._update_imu_msg(measurement)
        self._pub_imu_debug.publish(self._msg_imu_debug)
        # d) ground truth pose with covariance
        self._update_body_pose_with_cov_msg(measurement)
        self._pub_body_pose_with_cov.publish(self._msg_body_pose_with_cov)
        return

    """call backs"""

    def robot_simulation_step(self, step_size):
        """
        [Summary]

        Call robot update and advance, and tick ros bridge

        """
        self._a1.update()
        self._a1.advance()

        # Tick the ROS Clock
        og.Controller.evaluate_sync(self._clock_graph)

        # Publish ROS data
        self.publish_ros_data(self._a1._measurement)

    def joint_command_callback(self, data):
        """
        [Summary]

        Joint command call back, set command torque for the joints
        
        """
        for i in range(12):
            self._ros_command[i] = data.effort[i]

        self._a1.set_command_torque(self._ros_command)

    """
    Utilities functions.
    """

    def _update_body_pose_msg(self, measurement: A1Measurement):
        """
        [Summary]
        
        Updates the body pose message.
        
        """
        # base position
        self._msg_body_pose.pose.position.x = measurement.state.base_frame.pos[0]
        self._msg_body_pose.pose.position.y = measurement.state.base_frame.pos[1]
        self._msg_body_pose.pose.position.z = measurement.state.base_frame.pos[2]
        # base orientation
        self._msg_body_pose.pose.orientation.w = measurement.state.base_frame.quat[3]
        self._msg_body_pose.pose.orientation.x = measurement.state.base_frame.quat[0]
        self._msg_body_pose.pose.orientation.y = measurement.state.base_frame.quat[1]
        self._msg_body_pose.pose.orientation.z = measurement.state.base_frame.quat[2]

    def _update_msg_joint_state(self, measurement: A1Measurement):
        """
        [Summary]
        
        Updates the joint state message.
        
        """
        # joint position and velocity
        for i in range(12):
            self._msg_joint_state.position[i] = measurement.state.joint_pos[i]
            self._msg_joint_state.velocity[i] = measurement.state.joint_vel[i]
        # foot force
        for i in range(4):
            # notice this order is: FL, FR, RL, RR
            self._msg_joint_state.effort[12 + i] = measurement.foot_forces[i]

    def _update_imu_msg(self, measurement: A1Measurement):
        """
        [Summary]
        
        Updates the IMU message.
        
        """
        # accelerometer data
        self._msg_imu_debug.linear_acceleration.x = measurement.base_lin_acc[0]
        self._msg_imu_debug.linear_acceleration.y = measurement.base_lin_acc[1]
        self._msg_imu_debug.linear_acceleration.z = measurement.base_lin_acc[2]
        # gyroscope data
        self._msg_imu_debug.angular_velocity.x = measurement.base_ang_vel[0]
        self._msg_imu_debug.angular_velocity.y = measurement.base_ang_vel[1]
        self._msg_imu_debug.angular_velocity.z = measurement.base_ang_vel[2]

    def _update_body_pose_with_cov_msg(self, measurement: A1Measurement):
        """
        [Summary]
        
        Updates the body pose with fake covariance message.
        
        """
        # base position
        self._msg_body_pose_with_cov.pose.pose.position.x = measurement.state.base_frame.pos[0]
        self._msg_body_pose_with_cov.pose.pose.position.y = measurement.state.base_frame.pos[1]
        self._msg_body_pose_with_cov.pose.pose.position.z = measurement.state.base_frame.pos[2]
        # base orientation
        self._msg_body_pose_with_cov.pose.pose.orientation.w = measurement.state.base_frame.quat[3]
        self._msg_body_pose_with_cov.pose.pose.orientation.x = measurement.state.base_frame.quat[0]
        self._msg_body_pose_with_cov.pose.pose.orientation.y = measurement.state.base_frame.quat[1]
        self._msg_body_pose_with_cov.pose.pose.orientation.z = measurement.state.base_frame.quat[2]

        # Setting fake covariance
        for i in range(6):
            self._msg_body_pose_with_cov.pose.covariance[i * 6 + i] = 0.001


def main():
    """
    [Summary]

    The function launches the simulator, creates the robot, and run the simulation steps
    
    """
    # first enable ros node, make sure using simulation time
    rospy.init_node("isaac_a1", anonymous=False, disable_signals=True, log_level=rospy.ERROR)
    rospy.set_param("use_sim_time", True)
    physics_downtime = 1 / 400.0
    runner = A1_direct_runner(physics_dt=physics_downtime, render_dt=physics_downtime)
    simulation_app.update()
    runner.setup()

    # an extra reset is needed to register
    runner._world.reset()
    runner._world.reset()
    runner.run()
    rospy.signal_shutdown("a1 direct complete")
    simulation_app.close()


if __name__ == "__main__":
    main()
