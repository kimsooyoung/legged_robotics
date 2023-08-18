# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from dataclasses import field, dataclass
import numpy as np
from omni.isaac.quadruped.utils.types import NamedTuple, FrameState


@dataclass
class A1State(NamedTuple):
    """The kinematic state of the articulated robot."""

    base_frame: FrameState = field(default_factory=lambda: FrameState("root"))
    """State of base frame"""

    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(12))
    """Joint positions with shape: (12,)"""

    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(12))
    """Joint positions with shape: (12,)"""


@dataclass
class A1Measurement(NamedTuple):
    """The state of the robot along with the mounted sensor data."""

    state: A1State = field(default=A1State)
    """The state of the robot."""

    foot_forces: np.ndarray = field(default_factory=lambda: np.zeros(4))
    """Feet contact force of the robot in the order: FL, FR, RL, RR."""

    base_lin_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Accelerometer reading from IMU attached to robot's base."""

    base_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Gyroscope reading from IMU attached to robot's base."""


@dataclass
class A1Command(NamedTuple):
    """The command on the robot actuators."""

    desired_joint_torque: np.ndarray = field(default_factory=lambda: np.zeros(12))
    """Desired joint positions of the robot: (12,)"""
