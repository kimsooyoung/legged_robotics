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
class A1CtrlParams:
    """ A collection of parameters used by the QP agent """

    _robot_mass: float = field(default=16.0)
    """The mass of the robot"""

    _swap_foot_indices: np.array = field(default=np.array([1, 0, 3, 2], dtype=int))
    """A index list help to convert between A1 hardware leg index order and A1 Isaac Sim leg index order"""

    _foot_force_low: float = field(default=5.0)
    """ controller parameter: the low threshold of foot contact force"""

    _default_foot_pos: np.ndarray = field(
        default=np.array([[+0.17, +0.15, -0.3], [+0.17, -0.15, -0.3], [-0.17, +0.15, -0.3], [-0.17, -0.15, -0.3]])
    )
    """ controller parameter: the default foot pos in robot frame when the robot is standing still"""

    _kp_lin_x: float = field(default=0.0)
    """ control parameter: the raibert foothold strategy, x position target coefficient"""

    _kd_lin_x: float = field(default=0.15)
    """ control parameter: the raibert foothold strategy, x velocity target coefficient"""

    _kf_lin_x: float = field(default=0.2)
    """ control parameter: the raibert foothold strategy, x desired velocity target coefficient"""

    _kp_lin_y: float = field(default=0.0)
    """ control parameter: the raibert foothold strategy, y position target coefficient"""

    _kd_lin_y: float = field(default=0.1)
    """ control parameter: the raibert foothold strategy, y velocity target coefficient"""

    _kf_lin_y: float = field(default=0.2)
    """ control parameter: the raibert foothold strategy, y desired velocity target coefficient"""

    _kp_foot: np.ndarray = field(
        default=np.array(
            [[500.0, 500.0, 2000.0], [500.0, 500.0, 2000.0], [500.0, 500.0, 2000.0], [500.0, 500.0, 2000.0]]
        )
    )
    """ control parameter: the swing foot position error coefficient"""

    _kd_foot: np.ndarray = field(default=np.array([0.0, 0.0, 0.0]))
    """ control parameter: the swing foot velocity  error coefficient"""

    _km_foot: np.ndarray = field(default=np.diag([0.1, 0.1, 0.02]))
    """ control parameter: the swing foot force amplitude coefficient"""

    _kp_linear: np.ndarray = field(default=np.array([20.0, 20.0, 2000.0]))
    """ control parameter: the stance foot force position error coefficient"""

    _kd_linear: np.ndarray = field(default=np.array([50.0, 50.0, 0.0]))
    """ control parameter: the stance foot force velocity error coefficient"""

    _kp_angular: np.ndarray = field(default=np.array([600.0, 600.0, 10.0]))
    """ control parameter: the stance foot force orientation error coefficient"""

    _kd_angular: np.ndarray = field(default=np.array([3.0, 3.0, 10.0]))
    """ control parameter: the stance foot force orientation angular velocity error coefficient"""

    _torque_gravity: np.ndarray = field(default=np.array([0.80, 0, 0, -0.80, 0, 0, 0.80, 0, 0, -0.80, 0, 0]))
    """ control parameter: gravity compentation heuristic"""
