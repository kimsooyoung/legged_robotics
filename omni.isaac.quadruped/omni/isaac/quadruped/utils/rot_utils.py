# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


# python
import numba as nb
import numpy as np


@nb.jit(nopython=True)
def get_rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a rotation matrix.

    Args:
        quat (np.ndarray): A 4x1 vector in order (w, x, y, z)

    Returns:
        np.ndarray: The resulting 3x3 rotation matrix.
    """
    w, x, y, z = quat
    rot = np.array(
        [
            [2 * (w ** 2 + x ** 2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 2 * (w ** 2 + y ** 2) - 1, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w ** 2 + z ** 2) - 1],
        ]
    )
    return rot


@nb.jit(nopython=True)
def get_xyz_euler_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to XYZ euler angles.

    Args:
        quat (np.ndarray): A 4x1 vector in order (w, x, y, z).

    Returns:
        np.ndarray: A 3x1 vector containing (roll, pitch, yaw).
    """
    w, x, y, z = quat
    y_sqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y_sqr)
    eulerx = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    eulery = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y_sqr + z * z)
    eulerz = np.arctan2(t3, t4)

    result = np.zeros(3)
    result[0] = eulerx
    result[1] = eulery
    result[2] = eulerz

    return result


@nb.jit(nopython=True)
def get_quaternion_from_euler(euler: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """Convert an euler angle to a quaternion based on specified euler angle order.

    Supported Euler angle orders: {'XYZ', 'YXZ', 'ZXY', 'ZYX', 'YZX', 'XZY'}.

    Args:
        euler (np.ndarray): A 3x1 vector with angles in radians.
        order (str, optional): The specified order of input euler angles. Defaults to "XYZ".

    Raises:
        ValueError: If input order is not valid.

    Reference:
        [1] https://github.com/mrdoob/three.js/blob/master/src/math/Quaternion.js
    """
    # extract input angles
    r, p, y = euler
    # compute constants
    y = y / 2.0
    p = p / 2.0
    r = r / 2.0
    c3 = np.cos(y)
    s3 = np.sin(y)
    c2 = np.cos(p)
    s2 = np.sin(p)
    c1 = np.cos(r)
    s1 = np.sin(r)
    # convert to quaternion based on order
    if order == "XYZ":
        result = np.array(
            [
                c1 * c2 * c3 - s1 * s2 * s3,
                c1 * s2 * s3 + c2 * c3 * s1,
                c1 * c3 * s2 - s1 * c2 * s3,
                c1 * c2 * s3 + s1 * c3 * s2,
            ]
        )
        if result[0] < 0:
            result = -result
        return result
    elif order == "YXZ":
        result = np.array(
            [
                c1 * c2 * c3 + s1 * s2 * s3,
                s1 * c2 * c3 + c1 * s2 * s3,
                c1 * s2 * c3 - s1 * c2 * s3,
                c1 * c2 * s3 - s1 * s2 * c3,
            ]
        )
        return result
    elif order == "ZXY":
        result = np.array(
            [
                c1 * c2 * c3 - s1 * s2 * s3,
                s1 * c2 * c3 - c1 * s2 * s3,
                c1 * s2 * c3 + s1 * c2 * s3,
                c1 * c2 * s3 + s1 * s2 * c3,
            ]
        )
        return result
    elif order == "ZYX":
        result = np.array(
            [
                c1 * c2 * c3 + s1 * s2 * s3,
                s1 * c2 * c3 - c1 * s2 * s3,
                c1 * s2 * c3 + s1 * c2 * s3,
                c1 * c2 * s3 - s1 * s2 * c3,
            ]
        )
        return result
    elif order == "YZX":
        result = np.array(
            [
                c1 * c2 * c3 - s1 * s2 * s3,
                s1 * c2 * c3 + c1 * s2 * s3,
                c1 * s2 * c3 + s1 * c2 * s3,
                c1 * c2 * s3 - s1 * s2 * c3,
            ]
        )
        return result
    elif order == "XZY":
        result = np.array(
            [
                c1 * c2 * c3 + s1 * s2 * s3,
                s1 * c2 * c3 - c1 * s2 * s3,
                c1 * s2 * c3 - s1 * c2 * s3,
                c1 * c2 * s3 + s1 * s2 * c3,
            ]
        )
        return result
    else:
        raise ValueError("Input euler angle order is meaningless.")


@nb.jit(nopython=True)
def get_rotation_matrix_from_euler(euler: np.ndarray, order: str = "XYZ") -> np.ndarray:
    quat = get_quaternion_from_euler(euler, order)
    return get_rotation_matrix_from_quaternion(quat)


@nb.jit(nopython=True)
def quat_multiplication(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute the product of two quaternions.

    Args:
        q (np.ndarray): First quaternion in order (w, x, y, z).
        p (np.ndarray): Second quaternion in order (w, x, y, z).

    Returns:
        np.ndarray: A 4x1 vector representing a quaternion in order (w, x, y, z).
    """
    quat = np.array(
        [
            p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
            p[0] * q[1] + p[1] * q[0] - p[2] * q[3] + p[3] * q[2],
            p[0] * q[2] + p[1] * q[3] + p[2] * q[0] - p[3] * q[1],
            p[0] * q[3] - p[1] * q[2] + p[2] * q[1] + p[3] * q[0],
        ]
    )
    return quat


@nb.jit(nopython=True)
def skew(vector: np.ndarray) -> np.ndarray:
    """Convert vector to skew symmetric matrix.

    This function returns a skew-symmetric matrix to perform cross-product
    as a matrix multiplication operation, i.e.:

        np.cross(a, b) = np.dot(skew(a), b)


    Args:
        vector (np.ndarray): A 3x1 vector.

    Returns:
        np.ndarray: The resluting skew-symmetric matrix.
    """
    mat = np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])
    return mat
