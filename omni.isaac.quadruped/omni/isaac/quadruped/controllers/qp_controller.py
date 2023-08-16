# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Union, List
import numpy as np
import carb

# omni-isaac-a1
from omni.isaac.quadruped.utils.a1_classes import A1Measurement, A1Command

# QP controller related
from omni.isaac.quadruped.utils.a1_ctrl_states import A1CtrlStates
from omni.isaac.quadruped.utils.a1_ctrl_params import A1CtrlParams
from omni.isaac.quadruped.utils.a1_desired_states import A1DesiredStates
from omni.isaac.quadruped.controllers.a1_robot_control import A1RobotControl
from omni.isaac.quadruped.utils.a1_sys_model import A1SysModel
from omni.isaac.quadruped.utils.go1_sys_model import Go1SysModel
from omni.isaac.quadruped.utils.rot_utils import get_xyz_euler_from_quaternion, get_rotation_matrix_from_euler


class A1QPController:
    """[summary]

    A1 QP controller as a layer.

    An implementation of the QP controller[1]

    References:
        [1] Bledt, Gerardo, et al. "MIT Cheetah 3: Design and control of a robust, dynamic quadruped robot."
            2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018.
    """

    def __init__(self, name: str, _simulate_dt: float, waypoint_pose=None) -> None:
        """Initialize the QP Controller.

        Args:
            name {str} -- The name of the layer.
            _simulated_dt {float} -- rough estimation of the time interval of the control loop

        """
        # rough estimation of the time interval of the control loop
        self.simulate_dt = _simulate_dt

        # (nearly) constant control related parameters
        self._ctrl_params = A1CtrlParams()
        # control state varibles
        self._ctrl_states = A1CtrlStates()
        # control goal state varibles
        self._desired_states = A1DesiredStates()
        # robot controller
        self._root_control = A1RobotControl()
        # kinematic calculator
        if name == "A1":
            self._sys_model = A1SysModel()
        else:
            self._sys_model = Go1SysModel()

        # variables that toggle standing/moving mode
        self._init_transition = 0
        self._prev_transition = 0

        # an auto planner for collecting data
        self.waypoint_tgt_idx = 1
        if waypoint_pose is not None:
            self.waypoint_pose = waypoint_pose
        else:
            self.waypoint_pose = []

    """
    Operations
    """

    def setup(self) -> None:
        """[summary]

        Reset the ctrl states.
        """
        self.ctrl_state_reset()

    def reset(self) -> np.ndarray:
        """[summary]

        Reset the ctrl states.
        """
        self.ctrl_state_reset()

    def set_target_command(self, base_command: Union[List[float], np.ndarray]) -> None:
        """[summary]
        
        Set target base velocity command from joystick
        
        Args:
            base_command{Union[List[float], np.ndarray} -- velocity commands for the robot

        """
        self._current_base_command = base_command

    def advance(self, dt: float, measurement: A1Measurement, path_follow=False, auto_start=True) -> np.array:
        """[summary]
        
        Perform torque command generation.

        Args:
            dt {float} -- Timestep update in the world.
            measurement {A1Measurement} -- Current measurement from robot.
            path_follow {bool} -- True if a waypoint is pathed in, false if not
            auto_start {bool} -- True to start trotting after 1 second automatically, False for start trotting after "Enter" is pressed
        Returns:
            np.ndarray -- The desired joint torques for the robot.
        """
        # update controller states from A1Measurement
        self.update(dt, measurement)

        if auto_start:
            if (self._ctrl_states._exp_time > 1) and self._ctrl_states._init_transition == 0:
                self._ctrl_states._init_transition = 1

        # 간단한 경로 추적 
        if path_follow:
            if self._ctrl_states._exp_time > 6:
                if self.waypoint_tgt_idx == len(self.waypoint_pose) and self._ctrl_states._init_transition == 1:
                    self._ctrl_states._init_transition = 0
                    self._ctrl_states._prev_transition = 1
                    carb.log_info("stop motion")
                    self.waypoint_tgt_idx += 1

                elif self.waypoint_tgt_idx < len(self.waypoint_pose) and self._ctrl_states._init_transition == 1:
                    cur_pos = np.array(
                        [self._ctrl_states._root_pos[0], self._ctrl_states._root_pos[1], self._ctrl_states._euler[2]]
                    )

                    # position에서 x, y, yaw 만 빼낸다.
                    diff_pose = self.waypoint_pose[self.waypoint_tgt_idx] - cur_pos
                    diff_pos = np.array([diff_pose[0], diff_pose[1], 0])

                    # yaw angle 보정
                    # fix yaw angle for diff_pos
                    if diff_pose[2] > 1.5 * 3.14:  # tgt 3.14, cur -3.14
                        diff_pose[2] = diff_pose[2] - 6.28
                    if diff_pose[2] < -1.5 * 3.14:  # tgt -3.14, cur 3.14
                        diff_pose[2] = 6.28 + diff_pose[2]

                    # diff_pos를 body frame으로 변환한 뒤 아주 간단하게 * 10을 해서 경로로 전환한다.
                    # vel command body frame
                    diff_pos_r = self._ctrl_states._rot_mat_z.T @ diff_pos
                    self._current_base_command[0] = 10 * diff_pos_r[0]
                    self._current_base_command[1] = 10 * diff_pos_r[1]

                    # yaw command
                    self._current_base_command[2] = 10 * diff_pose[2]

                    # target pose에 도달하면 다음 target으로 넘어간다.
                    if np.linalg.norm(diff_pose) < 0.1 and self.waypoint_tgt_idx < len(self.waypoint_pose):
                        self.waypoint_tgt_idx += 1
                        # print(self.waypoint_tgt_idx, " - ", self.waypoint_pose[self.waypoint_tgt_idx])
                else:
                    # 모든 target pose에 도달했을 때
                    # self.waypoint_tgt_idx > len(self.waypoint_pose), in this case the planner is disabled
                    carb.log_info("target reached, back to manual control mode")
                    path_follow = False
                    pass

        # desired states update
        # velocity updates
        # update controller states from target command
        self._desired_states._root_lin_vel_d[0] = self._current_base_command[0]
        self._desired_states._root_lin_vel_d[1] = self._current_base_command[1]
        self._desired_states._root_ang_vel_d[2] = self._current_base_command[2]

        # euler angle update
        # _euler_d : desired body orientation in _euler angle
        self._desired_states._euler_d[2] += self._desired_states._root_ang_vel_d[2] * dt

        # position locking
        if self._ctrl_states._init_transition == 1:
            if np.linalg.norm(self._desired_states._root_lin_vel_d[0]) > 0.05:
                self._ctrl_params._kp_linear[0] = 0
                self._desired_states._root_pos_d[0] = self._ctrl_states._root_pos[0]
            if np.linalg.norm(self._desired_states._root_lin_vel_d[0]) < 0.05:
                self._ctrl_params._kp_linear[0] = 5000
            if np.linalg.norm(self._desired_states._root_lin_vel_d[1]) > 0.05:
                self._ctrl_params._kp_linear[1] = 0
                self._desired_states._root_pos_d[1] = self._ctrl_states._root_pos[1]
            if np.linalg.norm(self._desired_states._root_lin_vel_d[1]) < 0.05:
                self._ctrl_params._kp_linear[1] = 5000

            if np.linalg.norm(self._desired_states._root_ang_vel_d[2]) == 0:
                self._desired_states._euler_d[2] = self._ctrl_states._euler[2]

        # record position once when moving back into init transition = 0 state
        if self._ctrl_states._prev_transition == 1 and self._ctrl_states._init_transition < 1:
            self._ctrl_params._kp_linear[0:2] = np.array([500, 500])
            self._desired_states._euler_d[2] = self._ctrl_states._euler[2]
            self._desired_states._root_pos_d[0:2] = self._ctrl_states._root_pos[0:2]
            self._desired_states._root_lin_vel_d[0] = 0
            self._desired_states._root_lin_vel_d[1] = 0
            # make sure this logic only run once
            self._ctrl_states._prev_transition = self._ctrl_states._init_transition

        self._root_control.update_plan(self._desired_states, self._ctrl_states, self._ctrl_params, dt)

        # update_plan updates swing foot target
        # swing foot control and stance foot control
        torques = self._root_control.generate_ctrl(self._desired_states, self._ctrl_states, self._ctrl_params)
        return torques

    def switch_mode(self):
        """[summary]

        toggle between stationary/moving mode"""
        self._ctrl_states._prev_transition = self._ctrl_states._init_transition
        self._ctrl_states._init_transition = self._current_base_command[3]

    """
    Internal helpers.
    """

    def ctrl_state_reset(self) -> None:
        """[summary]
        
        reset _ctrl_states and _ctrl_params to non-default values
        """
        # following changes to A1CtrlParams alters the robot gait execution performance
        self._ctrl_params = A1CtrlParams()
        self._ctrl_params._kp_linear = np.array([500, 500.0, 1600.0])
        self._ctrl_params._kd_linear = np.array([2000.0, 2000.0, 4000.0])
        self._ctrl_params._kp_angular = np.array([600.0, 600.0, 0.0])
        self._ctrl_params._kd_angular = np.array([0.0, 0.0, 500.0])
        kp_foot_x = 11250.0
        kp_foot_y = 11250.0
        kp_foot_z = 11500.0
        self._ctrl_params._kp_foot = np.array(
            [
                [kp_foot_x, kp_foot_y, kp_foot_z],
                [kp_foot_x, kp_foot_y, kp_foot_z],
                [kp_foot_x, kp_foot_y, kp_foot_z],
                [kp_foot_x, kp_foot_y, kp_foot_z],
            ]
        )
        self._ctrl_params._kd_foot = np.array([0.0, 0.0, 0.0])
        self._ctrl_params._km_foot = np.diag([0.7, 0.7, 0.7])
        self._ctrl_params._robot_mass = 12.5
        self._ctrl_params._foot_force_low = 5.0

        self._ctrl_states = A1CtrlStates()
        self._ctrl_states._counter = 0.0
        self._ctrl_states._gait_counter = np.array([0.0, 0.0, 0.0, 0.0])
        self._ctrl_states._exp_time = 0.0

    def update(self, dt: float, measurement: A1Measurement):
        """[summary]
        
        Fill measurement into _ctrl_states
        Args:
            dt {float} -- Timestep update in the world.
            measurement {A1Measurement} -- Current measurement from robot.
        """
        self._ctrl_states._root_quat[0] = measurement.state.base_frame.quat[3]  # w
        self._ctrl_states._root_quat[1] = measurement.state.base_frame.quat[0]  # x
        self._ctrl_states._root_quat[2] = measurement.state.base_frame.quat[1]  # y
        self._ctrl_states._root_quat[3] = measurement.state.base_frame.quat[2]  # z

        self._ctrl_states._root_pos = measurement.state.base_frame.pos
        self._ctrl_states._root_lin_vel = measurement.state.base_frame.lin_vel

        if self._ctrl_states._root_quat[0] < 0:
            self._ctrl_states._root_quat = -self._ctrl_states._root_quat

        self._ctrl_states._euler = get_xyz_euler_from_quaternion(self._ctrl_states._root_quat)
        self._ctrl_states._rot_mat = get_rotation_matrix_from_euler(self._ctrl_states._euler)
        # according to rl_controler in isaac.anymal, base_frame.ang_vel is in world frame
        self._ctrl_states._root_ang_vel = self._ctrl_states._rot_mat.T @ measurement.state.base_frame.ang_vel
        self._ctrl_states._rot_mat_z = get_rotation_matrix_from_euler(np.array([0.0, 0.0, self._ctrl_states._euler[2]]))

        # still keep the option of using forward diff velocities
        for i in range(12):
            if abs(dt > 1e-10):
                self._ctrl_states._joint_vel[i] = (
                    measurement.state.joint_pos[i] - self._ctrl_states._joint_pos[i]
                ) / dt
            else:
                self._ctrl_states._joint_vel[i] = 0.0
            self._ctrl_states._joint_pos[i] = measurement.state.joint_pos[i]
            # self._ctrl_states._joint_vel[i] = measurement.state.joint_vel[i]

        for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
            # notice the id order of A1SysModel follows that on A1 hardware
            # [1, 0, 3, 2] -> [FL, FR, RL, RR]
            swap_i = self._ctrl_params._swap_foot_indices[i]
            self._ctrl_states._foot_pos_rel[i, :] = self._sys_model.forward_kinematics(
                swap_i, self._ctrl_states._joint_pos[i * 3 : (i + 1) * 3]
            )
            self._ctrl_states._j_foot[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = self._sys_model.jacobian(
                swap_i, self._ctrl_states._joint_pos[i * 3 : (i + 1) * 3]
            )

            self._ctrl_states._foot_pos_abs[i, :] = self._ctrl_states._rot_mat @ self._ctrl_states._foot_pos_rel[i, :]

            self._ctrl_states._foot_forces[i] = measurement.foot_forces[i]

        self._ctrl_states._exp_time += dt
