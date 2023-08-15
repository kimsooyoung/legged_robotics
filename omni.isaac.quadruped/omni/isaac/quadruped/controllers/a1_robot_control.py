# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


import numpy as np

# bezier is used in leg trajectory generation
import bezier

# use osqp to solve QP
import osqp
import scipy.sparse as sp
from typing import Tuple

from omni.isaac.quadruped.utils.rot_utils import skew
from omni.isaac.quadruped.utils.a1_ctrl_states import A1CtrlStates
from omni.isaac.quadruped.utils.a1_ctrl_params import A1CtrlParams
from omni.isaac.quadruped.utils.a1_desired_states import A1DesiredStates


class A1RobotControl:
    """[summary]

    The A1 robot controller
    This class uses A1CtrlStates to save data. The control joint torque is generated
    using a QP controller
    """

    def __init__(self) -> None:
        """Initializes the class instance.
        """
        pass

    """
    Operations
    """

    def update_plan(
        self, desired_states: A1DesiredStates, input_states: A1CtrlStates, input_params: A1CtrlParams, dt: float
    ) -> None:
        """[summary]
        
        update swing leg trajectory and several counters

        Args:
            desired_states {A1DesiredStates} -- the desired states
            input_states {A1CtrlStates} -- the control states
            input_params {A1CtrlParams}     -- the control parameters
            dt {float} -- The simulation time-step.

        """
        self._update_gait_plan(input_states)
        self._update_foot_plan(desired_states, input_states, input_params, dt)
        # increase _counter
        input_states._counter += 1
        input_states._exp_time += dt

        input_states._gait_counter += input_states._gait_counter_speed
        input_states._gait_counter %= input_states._counter_per_gait

    def generate_ctrl(
        self, desired_states: A1DesiredStates, input_states: A1CtrlStates, input_params: A1CtrlParams
    ) -> None:
        """ [summary]
        
        main function, generate foot ground reaction force using QP and calculate joint torques

        Args:
            desired_states {A1DesiredStates} -- the desired states
            input_states {A1CtrlStates} -- the control states
            input_params {A1CtrlParams} -- the control parameters
        """
        # first second, do nothing, wait sensor and stuff got stablized
        if input_states._exp_time < 0.1:
            return np.zeros(12)
        # initial control
        if input_states._init_transition == 0 and input_states._prev_transition == 0:
            input_params._kp_linear[0:2] = np.array([500, 500])

        # foot control
        foot_pos_final = input_states._foot_pos_target_rel
        foot_pos_cur = np.zeros([4, 3])

        for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
            # robot frame으로 변환
            foot_pos_cur[i, :] = input_states._rot_mat_z.T @ input_states._foot_pos_abs[i, :]

        bezier_time = np.zeros(4)
        for i in range(4):
            if input_states._gait_counter[i] < input_states._counter_per_swing:
                bezier_time[i] = 0.0
                input_states._foot_pos_start_rel[i, :] = foot_pos_cur[i, :]
                input_states._early_contacts[i] = False
            else:
                bezier_time[i] = (
                    input_states._gait_counter[i] - input_states._counter_per_swing
                ) / input_states._counter_per_swing

        # _rot_mat_z : R^{world}_{robot yaw}
        # 
        # _foot_pos_start_rel : (R^{world}_{robot yaw}).T * _foot_pos_abs(rotated robot frame) 
        #                       => robot frame 기준 현재 다리 위치
        # foot_pos_final : _foot_pos_target_rel (foot target pos in the relative frame (robot frame))
        # 
        # foot_pos_target => robot frame 기준 target
        foot_pos_target = self._get_from_bezier_curve(input_states._foot_pos_start_rel, foot_pos_final, bezier_time)
        foot_pos_error = foot_pos_target - foot_pos_cur
        foot_forces_kin = (input_params._kp_foot * foot_pos_error).flatten()

        # detect early contacts
        # how to determine which foot is in contact: check gait counter
        for i in range(4):
            if not input_states._contacts[i] and input_states._gait_counter[i] <= input_states._counter_per_swing * 1.5:
                input_states._early_contacts[i] = False
            if (
                not input_states._contacts[i]
                and input_states._early_contacts[i] is False
                and input_states._foot_forces[i] > input_params._foot_force_low
                and input_states._gait_counter[i] > input_states._counter_per_swing * 1.5
            ):
                input_states._early_contacts[i] = True

        for i in range(4):
            input_states._contacts[i] = input_states._contacts[i] or input_states._early_contacts[i]

        # root control

        grf = self._compute_grf(desired_states, input_states, input_params)
        grf_rel = grf @ input_states._rot_mat
        foot_forces_grf = -grf_rel.flatten()

        # convert to torque
        M = np.kron(np.eye(4, dtype=int), input_params._km_foot)
        # torques_init = input_states._j_foot.T @ foot_forces_init
        torques_kin = np.linalg.inv(input_states._j_foot) @ M @ foot_forces_kin
        # torques_kin = input_states._j_foot.T @ foot_forces_kin
        torques_grf = input_states._j_foot.T @ foot_forces_grf

        # combine torques
        torques_init = np.zeros(12)
        for i in range(4):
            torques_init[3 * i : 3 * i + 3] = torques_grf[3 * i : 3 * i + 3]

        # combine torques
        torques = np.zeros(12)
        for i in range(4):
            if input_states._contacts[i]:
                torques[3 * i : 3 * i + 3] = torques_grf[3 * i : 3 * i + 3]
            else:
                torques[3 * i : 3 * i + 3] = torques_kin[3 * i : 3 * i + 3]

        torques = (1 - input_states._init_transition) * torques_init + input_states._init_transition * torques
        torques += input_params._torque_gravity

        # for i in range(12):
        #     if torques[i] < -1000:
        #         torques[i] = -1000
        #     if torques[i] > 1000:
        #         torques[i] = 1000

        return torques

    """
    Internal helpers.
    """

    def _update_gait_plan(self, input_states: A1CtrlStates) -> None:
        """ [summary]
        
        update gait counters

        Args:
            input_states {A1CtrlStates} -- the control states
        """

        # initialize _counter
        if input_states._counter == 0 or input_states._gait_type != input_states._gait_type_last:
            if input_states._gait_type == 2:
                input_states._gait_counter = np.array([0.0, 120.0, 0.0, 120.0])
            elif input_states._gait_type == 1:
                input_states._gait_counter = np.array([0.0, 120.0, 120.0, 0.0])
            else:
                input_states._gait_counter = np.array([0.0, 0.0, 0.0, 0.0])

        # update _counter speed
        for i in range(4):
            if input_states._gait_type == 2:
                input_states._gait_counter_speed[i] = 1.4
            elif input_states._gait_type == 1:
                input_states._gait_counter_speed[i] = 1.4
            else:
                input_states._gait_counter_speed[i] = 0.0

            input_states._contacts[i] = input_states._gait_counter[i] < input_states._counter_per_swing

        input_states._gait_type_last = input_states._gait_type

    def _update_foot_plan(
        self, desired_states: A1DesiredStates, input_states: A1CtrlStates, input_params: A1CtrlParams, dt: float
    ) -> None:
        """ [summary]

        update foot swing target positions

        Args:
            input_states {A1DesiredStates} -- the desried states
            input_states {A1CtrlStates}    -- the control states
            input_params {A1CtrlParams}    -- the control parameters
            dt           {float}           -- delta time since last update
        """

        # heuristic plan
        lin_pos = input_states._root_pos
        # lin_pos_rel = input_states._rot_mat_z.T @ lin_pos

        lin_pos_d = desired_states._root_pos_d
        # lin_pos_rel_d = input_states._rot_mat_z.T @ lin_pos_d

        lin_vel = input_states._root_lin_vel
        # body frame
        lin_vel_rel = input_states._rot_mat_z.T @ lin_vel

        input_states._foot_pos_target_rel = input_params._default_foot_pos.copy()
        for i in range(4):
            weight_y = np.square(np.abs(input_params._default_foot_pos[i, 2]) / 9.8)
            weight2 = input_states._counter_per_swing / input_states._gait_counter_speed[i] * dt / 2.0
            delta_x = weight_y * (lin_vel_rel[0] - desired_states._root_lin_vel_d[0]) + weight2 * lin_vel_rel[0]
            delta_y = weight_y * (lin_vel_rel[1] - desired_states._root_lin_vel_d[1]) + weight2 * lin_vel_rel[1]

            if delta_x < -0.1:
                delta_x = -0.1
            if delta_x > 0.1:
                delta_x = 0.1
            if delta_y < -0.1:
                delta_y = -0.1
            if delta_y > 0.1:
                delta_y = 0.1

            input_states._foot_pos_target_rel[i, 0] += delta_x
            input_states._foot_pos_target_rel[i, 1] += delta_y

    def _get_from_bezier_curve(
        self, foot_pos_start: np.ndarray, foot_pos_final: np.ndarray, bezier_time: float
    ) -> np.ndarray:
        """[summary]

        generate swing foot position target from a bezier curve

        Args:
            foot_pos_start {np.ndarray} -- The curve start point
            foot_pos_final {np.ndarray} -- The curve end point
            bezier_time {float} -- The curve interpolation time, should be within [0,1].

        
        """
        bezier_degree = 4
        bezier_s = np.linspace(0, 1, bezier_degree + 1)
        bezier_nodes = np.zeros([2, bezier_degree + 1])

        bezier_nodes[0, :] = bezier_s

        foot_pos_target = np.zeros([4, 3])
        foot_pos_target_x = foot_pos_target[:, 0]
        foot_pos_target_y = foot_pos_target[:, 1]
        foot_pos_target_z = foot_pos_target[:, 2]

        for i in range(4):
            bezier_x = np.array(
                [
                    foot_pos_start[i, 0],
                    foot_pos_start[i, 0],
                    foot_pos_final[i, 0],
                    foot_pos_final[i, 0],
                    foot_pos_final[i, 0],
                ]
            )
            bezier_nodes[1, :] = bezier_x
            bezier_curve = bezier.Curve(bezier_nodes, bezier_degree)
            foot_pos_target_x[i] = bezier_curve.evaluate(bezier_time[i])[1, 0]

        for i in range(4):
            bezier_y = np.array(
                [
                    foot_pos_start[i, 1],
                    foot_pos_start[i, 1],
                    foot_pos_final[i, 1],
                    foot_pos_final[i, 1],
                    foot_pos_final[i, 1],
                ]
            )
            bezier_nodes[1, :] = bezier_y
            bezier_curve = bezier.Curve(bezier_nodes, bezier_degree)
            foot_pos_target_y[i] = bezier_curve.evaluate(bezier_time[i])[1, 0]

        for i in range(4):
            bezier_z = np.array(
                [
                    foot_pos_start[i, 2],
                    foot_pos_start[i, 2],
                    foot_pos_final[i, 2],
                    foot_pos_final[i, 2],
                    foot_pos_final[i, 2],
                ]
            )
            foot_clearance1 = 0.0
            foot_clearance2 = 0.5
            bezier_z[1] += foot_clearance1
            bezier_z[2] += foot_clearance2

            bezier_nodes[1, :] = bezier_z
            bezier_curve = bezier.Curve(bezier_nodes, bezier_degree)
            foot_pos_target_z[i] = bezier_curve.evaluate(bezier_time[i])[1, 0]

        return foot_pos_target

    def _compute_grf(
        self, desired_states: A1DesiredStates, input_states: A1CtrlStates, input_params: A1CtrlParams
    ) -> np.ndarray:
        """ [summary]
        
        main internal function, generate foot ground reaction force using QP

        Args:
            desired_states {A1DesiredStates} -- the desired states
            input_states {A1CtrlStates} -- the control states
            input_params {A1CtrlParams}     -- the control parameters

        Returns:
            grf {np.ndarray}
        """

        inertia_inv, root_acc, acc_weight, u_weight = self._get_qp_params(desired_states, input_states, input_params)

        modified_contacts = np.array([True, True, True, True])
        if input_states._init_transition < 1.0:
            modified_contacts = np.array([True, True, True, True])
        else:
            modified_contacts = input_states._contacts

        mu = 0.2
        # use osqp
        # np.diag(np.square(np.array([1, 1, 1, 20, 20, 10])))
        # array([[  1,   0,   0,   0,   0,   0],
        #     [  0,   1,   0,   0,   0,   0],
        #     [  0,   0,   1,   0,   0,   0],
        #     [  0,   0,   0, 400,   0,   0],
        #     [  0,   0,   0,   0, 400,   0],
        #     [  0,   0,   0,   0,   0, 100]])
        # u_weight = 1e-3
        Q = np.diag(np.square(acc_weight))
        R = u_weight
        F_min = 0
        F_max = 250.0
        hessian = np.identity(12) * R + inertia_inv.T @ Q @ inertia_inv
        gradient = -inertia_inv.T @ Q @ root_acc
        linearMatrix = np.zeros([20, 12])
        lowerBound = np.zeros(20)
        upperBound = np.zeros(20)
        for i in range(4):
            # extract F_zi
            linearMatrix[i, 2 + i * 3] = 1.0
            # friction pyramid
            # 1. F_xi < uF_zi
            linearMatrix[4 + i * 4, i * 3] = 1.0
            linearMatrix[4 + i * 4, 2 + i * 3] = -mu
            lowerBound[4 + i * 4] = -np.inf
            # 2. -F_xi > uF_zi
            linearMatrix[4 + i * 4 + 1, i * 3] = -1.0
            linearMatrix[4 + i * 4 + 1, 2 + i * 3] = -mu
            lowerBound[4 + i * 4 + 1] = -np.inf
            # 3. F_yi < uF_zi
            linearMatrix[4 + i * 4 + 2, 1 + i * 3] = 1.0
            linearMatrix[4 + i * 4 + 2, 2 + i * 3] = -mu
            lowerBound[4 + i * 4 + 2] = -np.inf
            # 4. -F_yi > uF_zi
            linearMatrix[4 + i * 4 + 3, 1 + i * 3] = -1.0
            linearMatrix[4 + i * 4 + 3, 2 + i * 3] = -mu
            lowerBound[4 + i * 4 + 3] = -np.inf

            c_flag = 1.0 if modified_contacts[i] else 0.0
            lowerBound[i] = c_flag * F_min
            upperBound[i] = c_flag * F_max

        sparse_hessian = sp.csc_matrix(hessian)

        # initialize the OSQP solver
        solver = osqp.OSQP()
        solver.setup(
            P=sparse_hessian, q=gradient, A=sp.csc_matrix(linearMatrix), l=lowerBound, u=upperBound, verbose=False
        )
        results = solver.solve()
        # print("compare casadi with osqp")
        # print(grf_vec)
        # print(results.x)

        grf = results.x.reshape(4, 3)
        # print(results.x)
        # print(grf)
        return grf

    def _get_qp_params(
        self, desired_states: A1DesiredStates, input_states: A1CtrlStates, input_params: A1CtrlParams
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ [summary]
        main internal function, construct parameters of the QP problem

        Args:
            desired_states {A1DesiredStates} -- the desired states
            input_states {A1CtrlStates} -- the control states
            input_params {A1CtrlParams} -- the control parameters

        Returns:
            qp_params: {Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- inertia_inv, root_acc, acc_weight, u_weight}
        """

        # continuous yaw error
        # reference: http://ltu.diva-portal.org/smash/get/diva2:1010947/FULLTEXT01.pdf
        euler_error = desired_states._euler_d - input_states._euler

        # _euler_d : the desired body orientation in _euler angle
        # _euler : robot _euler angle in world frame

        # limit euler error to pi/2
        if euler_error[2] > 3.1415926 * 1.5:  # eulerd 3.14 euler -3.14
            euler_error[2] = desired_states._euler_d[2] - 3.1415926 * 2 - input_states._euler[2]
            # euler_error[2] = euler_error[2] - 3.1415926 * 2
        elif euler_error[2] < -3.1415926 * 1.5:
            euler_error[2] = desired_states._euler_d[2] + 3.1415926 * 2 - input_states._euler[2]

        root_acc = np.zeros(6)
        # _root_pos_d : the desired body position in world frame
        # _root_pos : robot position in world frame
        root_acc[0:3] = input_params._kp_linear * (desired_states._root_pos_d - input_states._root_pos)

        # 결국 다 world로 바꾼다.
        # _root_lin_vel_d이 robot frame 기준이어서 _root_lin_vel를 다시 robot frame으로 바꾼 뒤 다시 변환하는 것임
        # _rot_mat_z : R^{world}_{robot yaw}
        # _root_lin_vel_d : the desired body velocity in robot frame
        # _root_lin_vel : robot linear velocity in world frame
        root_acc[0:3] += input_states._rot_mat_z @ (
            input_params._kd_linear
            * (desired_states._root_lin_vel_d - input_states._rot_mat_z.T @ input_states._root_lin_vel)
        )

        # euler_error도 world 기준임
        # _root_ang_vel_d : the desired body angular velocity 
        # _root_ang_vel : robot angular velocity in world frame TODO: 여기 조금 이상하네 주석이 잘못된건가?
        root_acc[3:6] = input_params._kp_angular * euler_error
        root_acc[3:6] += input_params._kd_angular * (
            desired_states._root_ang_vel_d - input_states._rot_mat_z.T @ input_states._root_ang_vel
        )

        # Add gravity
        mass = input_params._robot_mass
        root_acc[2] += mass * 9.8

        for i in range(6):
            if root_acc[i] < -500:
                root_acc[i] = -500
            if root_acc[i] > 500:
                root_acc[i] = 500

        # Create inverse inertia matrix
        # 
        inertia_inv = np.zeros([6, 12])
        inertia_inv[0:3] = np.tile(np.eye(3), 4)  # TODO: use the real inertia from URDF
        for i in range(4):
            skew_mat = skew(input_states._foot_pos_abs[i, :])
            inertia_inv[3:6, i * 3 : i * 3 + 3] = input_states._rot_mat_z.T @ skew_mat

        # QP weight
        acc_weight = np.array([1, 1, 1, 20, 20, 10])
        u_weight = 1e-3

        return inertia_inv, root_acc, acc_weight, u_weight
