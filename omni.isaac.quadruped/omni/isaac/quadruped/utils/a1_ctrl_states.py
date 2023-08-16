# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
from dataclasses import dataclass, field


@dataclass
class A1CtrlStates:
    """ A collection of variables used by the QP agent """

    _counter_per_gait: float = field(default=240.0)
    """The number of ticks of one gait cycle"""

    _counter_per_swing: float = field(default=120.0)
    """The number of ticks of one swing phase (half of the gait cycle)"""

    _counter: float = field(default=0.0)
    """A _counter used to determine how many ticks since the simulation starts"""

    _exp_time: float = field(default=0.0)
    """Simulation time since the simulation starts"""

    _gait_counter: np.array = field(default_factory=lambda: np.zeros(4))
    """Each leg has its own _counter with initial phase"""

    _gait_counter_speed: np.array = field(default_factory=lambda: np.zeros(4))
    """The speed of gait _counter update"""

    _root_pos: np.array = field(default_factory=lambda: np.zeros(3))
    """feedback state: robot position in world frame"""

    _root_quat: np.array = field(default_factory=lambda: np.zeros(4))
    """feedback state: robot quaternion in world frame"""

    _root_lin_vel: np.array = field(default_factory=lambda: np.zeros(3))
    """feedback state: robot linear velocity in world frame"""

    _root_ang_vel: np.array = field(default_factory=lambda: np.zeros(3))
    """feedback state: robot angular velocity in world frame"""
    # 오타 같은데, robot frame 같음

    _joint_pos: np.array = field(default_factory=lambda: np.zeros(12))
    """feedback state: robot motor joint angles"""

    _joint_vel: np.array = field(default_factory=lambda: np.zeros(12))
    """feedback state: robot motor joint angular velocities"""

    _foot_forces: np.array = field(default_factory=lambda: np.zeros(4))
    """feedback state: robot foot contact forces"""

    _foot_pos_target_world: np.ndarray = field(default_factory=lambda: np.zeros([4, 3]))
    """ controller variables: the foot target pos in the world frame"""

    _foot_pos_target_abs: np.ndarray = field(default_factory=lambda: np.zeros([4, 3]))
    """ controller variables: the foot target pos in the absolute frame (rotated robot frame)"""

    _foot_pos_target_rel: np.ndarray = field(default_factory=lambda: np.zeros([4, 3]))
    """ controller variables: the foot target pos in the relative frame (robot frame)"""

    _foot_pos_start_rel: np.ndarray = field(default_factory=lambda: np.zeros([4, 3]))
    """ controller variables: the foot current pos in the relative frame (robot frame)"""

    _euler: np.array = field(default_factory=lambda: np.zeros(3))
    """indirect feedback state: robot _euler angle in world frame"""

    _rot_mat: np.ndarray = field(default_factory=lambda: np.zeros([3, 3]))
    """indirect feedback state: robot rotation matrix in world frame"""

    _rot_mat_z: np.ndarray = field(default_factory=lambda: np.zeros([3, 3]))
    """indirect feedback state: robot rotation matrix with just the yaw angle in world frame"""
    # R^{world}_{robot yaw}

    _foot_pos_abs: np.ndarray = field(default_factory=lambda: np.zeros([4, 3]))
    """ controller variables: the foot current pos in the absolute frame (rotated robot frame)"""

    _foot_pos_rel: np.ndarray = field(default_factory=lambda: np.zeros([4, 3]))
    """ controller variables: the foot current pos in the relative frame (robot frame)"""

    _j_foot: np.ndarray = field(default_factory=lambda: np.zeros([12, 12]))
    """ controller variables: the foot jacobian in the relative frame (robot frame)"""

    _gait_type: int = field(default=1)
    """ control variable: type of gait, currently only 1 is defined, which is a troting gait"""

    _gait_type_last: int = field(default=1)
    """ control varialbe: saves the previous gait. Reserved for future use"""

    _contacts: np.array = field(default_factory=lambda: np.array([False] * 4))
    """ control varialbe: determine each foot has contact with ground or not"""

    _early_contacts: np.array = field(default_factory=lambda: np.array([False] * 4))
    """ control varialbe: determine each foot has early contact with ground or not (unexpect contact during foot swing)"""

    _init_transition: int = field(default=0)
    """ control variable: determine whether the robot should be in walking mode or standstill mode """
    _prev_transition: int = field(default=0)
    """ control variable: previous mode"""
