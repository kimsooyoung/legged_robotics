# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""[summary]

The kinematics parameters value come from  
https://github.com/unitreerobotics/unitree_ros/blob/master/robots/a1_description/xacro/const.xacro

It calculates the forward kinematics and jacobians of the Unitree A1 robot legs 
"""
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Go1SysModel:
    """Constants and functions related to the forward kinematics of the robot"""

    """
    Properties
    """
    THIGH_OFFSET = 0.08
    """constant: the length of the thigh motor"""

    LEG_OFFSET_X = 0.1881
    """constant: x distance from the robot COM to the leg base"""

    LEG_OFFSET_Y = 0.04675
    """constant: y distance from the robot COM to the leg base"""

    THIGH_LENGTH = 0.213
    """constant: length of the leg"""

    C_FR = 0
    """constant: FR leg id in A1's hardware convention"""

    C_FL = 1
    """constant: FL leg id in A1's hardware convention"""

    C_RR = 2
    """constant: RR leg id in A1's hardware convention"""

    C_RL = 3
    """constant: RL leg id in A1's hardware convention"""

    def __init__(self):
        """Initializes the class instance.
        """
        pass

    """
    Operations
    """

    def forward_kinematics(self, idx: int, q: np.array) -> np.array:
        """get the forward_kinematics of the leg

        Arguments:
            idx {int}: the index of the leg, must use the A1 hardware convention
            q {np.array}: the joint angles of a leg
        """
        # these two variables indicates the quadrant of the leg
        fx = self.LEG_OFFSET_X
        fy = self.LEG_OFFSET_Y
        d = self.THIGH_OFFSET
        if idx == self.C_FR:
            fy *= -1
            d *= -1
        elif idx == self.C_FL:
            pass
        elif idx == self.C_RR:
            fx *= -1
            fy *= -1
            d *= -1
        else:
            fx *= -1

        length = self.THIGH_LENGTH
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        p = np.zeros(3)
        p[0] = fx - length * np.sin(q2 + q3) - length * np.sin(q2)
        p[1] = (
            fy
            + d * np.cos(q1)
            + length * np.cos(q2) * np.sin(q1)
            + length * np.cos(q2) * np.cos(q3) * np.sin(q1)
            - length * np.sin(q1) * np.sin(q2) * np.sin(q3)
        )
        p[2] = (
            d * np.sin(q1)
            - length * np.cos(q1) * np.cos(q2)
            - length * np.cos(q1) * np.cos(q2) * np.cos(q3)
            + length * np.cos(q1) * np.sin(q2) * np.sin(q3)
        )
        return p

    def jacobian(self, idx: int, q: np.array) -> np.ndarray:
        """get the jacobian of the leg

        Arguments:
            idx {int}: the index of the leg, must use the A1 hardware convention
            q {np.array}: the joint angles of a leg
        """
        # these two variables indicates the quadrant of the leg
        fx = self.LEG_OFFSET_X
        fy = self.LEG_OFFSET_Y
        d = self.THIGH_OFFSET
        if idx == self.C_FR:
            fy *= -1
            d *= -1
        elif idx == self.C_FL:
            pass
        elif idx == self.C_RR:
            fx *= -1
            fy *= -1
            d *= -1
        else:
            fx *= -1

        length = self.THIGH_LENGTH
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        J = np.zeros([3, 3])
        # J[1,1] = 0
        J[0, 1] = -length * (np.cos(q2 + q3) + np.cos(q2))
        J[0, 2] = -length * np.cos(q2 + q3)
        J[1, 0] = (
            length * np.cos(q1) * np.cos(q2)
            - d * np.sin(q1)
            + length * np.cos(q1) * np.cos(q2) * np.cos(q3)
            - length * np.cos(q1) * np.sin(q2) * np.sin(q3)
        )
        J[1, 1] = -length * np.sin(q1) * (np.sin(q2 + q3) + np.sin(q2))
        J[1, 2] = -length * np.sin(q2 + q3) * np.sin(q1)
        J[2, 0] = (
            d * np.cos(q1)
            + length * np.cos(q2) * np.sin(q1)
            + length * np.cos(q2) * np.cos(q3) * np.sin(q1)
            - length * np.sin(q1) * np.sin(q2) * np.sin(q3)
        )
        J[2, 1] = length * np.cos(q1) * (np.sin(q2 + q3) + np.sin(q2))
        J[2, 2] = length * np.sin(q2 + q3) * np.cos(q1)
        return J

    def foot_vel(self, idx: int, q: np.array, dq: np.array) -> np.array:
        """get the foot velocity

        Arguments:
            idx {int}: the index of the leg, must use the A1 hardware convention
            q {np.array}: the joint angles of a leg
            dq {np.array}: the joint angular velocities of a leg
        """
        my_jacobian = self.jacobian(idx, q)
        vel = my_jacobian @ dq
        return vel
