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
class A1DesiredStates:
    """ A collection of desired goal states used by the QP agent """

    _root_pos_d: np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.35]))
    """ control goal paramter: the desired body position in world frame"""

    _root_lin_vel_d: np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    """ control goal paramter: the desired body velocity in robot frame """

    _euler_d: np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    """ control goal paramter: the desired body orientation in _euler angle """

    _root_ang_vel_d: np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    """ control goal paramter: the desired body angular velocity """
