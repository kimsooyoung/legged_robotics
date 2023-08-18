# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


# enable deferred annotations
from __future__ import annotations

# python
import dataclasses
from dataclasses import dataclass, field
from typing import List, Union, Dict, Any
import numpy as np


@dataclass
class NamedTuple(object):
    """[Summary]
    
    The backend data structure for data-passing between various modules.

    In order to support use cases where the data would have mixed types (such as bool/integer/array), we provide a
    light data-class to capture this formalism while allowing the data to be shared between different modules easily.
    The objective is to support complex agent designs and support multi-agent environments.

    The usage of this class is quite similar to that of a dictionary, since underneath, we rely on the key names to
    "pass" data from one container into another. However, we do not use the dictionary since a data-class helps in
    providing type hints which is in practice quite useful.

    Reference:
        https://stackoverflow.com/questions/51671699/data-classes-vs-typing-namedtuple-primary-use-cases

    """

    def update(self, data: Union[NamedTuple, List[NamedTuple], Dict[str, Any]]):
        """Update values from another named tuple.

        Note:
            Unlike `dict.update(dict)`, this method does not add element(s) to the instance if the key is not present.

        Arguments:
            data {Union[NamedTuple, List[NamedTuple], Dict[str, Any]} -- The input data to update values from.

        Raises:
            TypeError -- When input data is not of type :class:`NamedTuple` or :class:`List[NamedTuple]`.
        """
        # convert to dictionary
        if isinstance(data, dict):
            data_dict = data
        elif isinstance(data, list):
            data_dict = {}
            for d in data:
                data_dict.update(d.__dict__)
        elif isinstance(data, NamedTuple):
            data_dict = data.__dict__
        else:
            name = self.__class__.__name__
            raise TypeError(
                f"Invalid input data type: {type(data)}. Valid: [`{name}`, `List[{name}]`, `Dict[str, Any]`]."
            )
        # iterate over dictionary and add values to matched keys
        for key, value in data_dict.items():
            try:
                self.__setattr__(key, value)
            except AttributeError:
                pass

    def as_dict(self) -> dict:
        """Converts the dataclass to dictionary recursively.

        Returns:
            dict: Instance information as a dictionary
        """
        return dataclasses.asdict(self)


@dataclass
class FrameState(NamedTuple):
    """The state of a kinematic frame.

    Attributes:
        name: The name of the frame.
        pos: The Cartesian position of the frame.
        quat: The quaternion orientation (x, y, z, w) of the frame.
        lin_vel: The linear velocity of the frame.
        ang_vel: The angular velocity of the frame.
    """

    name: str
    """Frame name."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Catersian position of frame."""

    quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    """Quaternion orientation of frame: (x, y, z, w)"""

    lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Linear velocity of frame."""

    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Angular velocity of frame."""

    @property
    def pose(self) -> np.ndarray:
        """Returns: A numpy array with position and orientation."""
        return np.concatenate([self.pos, self.quat])


# EOF
