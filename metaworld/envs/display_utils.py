import numpy as np
import matplotlib.colors

import warnings

COLOR_LIST = [
    "#F27970",
    "#BB9727",
    "#54B345",
    "#32B897",
    "#05B9E2",
    "#8983BF",
    "#C76DA2",
]
RGB_COLOR_LIST = [matplotlib.colors.to_rgb(color) for color in COLOR_LIST]

QUAT_LIST = [
    [1., 0., 0., 0.],                               # 正常状态
    [np.sin(np.pi/4), 0., 0., np.sin(np.pi/4)],     # 俯视顺时针旋转90
    [np.sin(np.pi/4), 0., 0., -np.sin(np.pi/4)],    # 俯视逆时针旋转90
    [0., 0., 0., 1.],                               # 俯视旋转180
]


def random_grid_pos(x_range, y_range, forbid_list=None):
    if forbid_list is None:
        forbid_list = []
    while True:
        x = np.random.randint(int(np.round(x_range[0]*100)), int(np.round(x_range[1]*100))) / 100
        y = np.random.randint(int(np.round(y_range[0]*100)), int(np.round(y_range[1]*100))) / 100
        flag = True
        for (min_x, max_x), (min_y, max_y) in forbid_list:
            if min_x <= x <= max_x and min_y <= y <= max_y:
                flag = False
                break
        if flag:
            break
    return x, y


def obstacle_in_path(pos_curr, pos_targ, pos_obst):
    """
    Return True if there is an (tall) obstacle (e.g., coffee machine)
    in the path from current to target position.
    
    Args:
        pos_curr: current position.
        pos_targ: target position.
        pos_obst: obstacle position.
    """
    return (abs(pos_curr[0] - pos_targ[0]) > abs(pos_obst[0] - pos_targ[0])) \
            and ((pos_curr[0] - pos_targ[0]) * (pos_obst[0] - pos_targ[0]) > 0)


def near_obstacle(pos_curr, pos_obst, tolerance=0.2):
    """Return True if hand near obstacle"""
    return np.linalg.norm(pos_curr[:2] - pos_obst[:2]) < tolerance

def safe_move(from_xyz, to_xyz, p):
    """Computes action components that help move from 1 position to another

    Args:
        from_xyz (np.ndarray): The coordinates to move from (usually current position)
        to_xyz (np.ndarray): The coordinates to move to
        p (float): constant to scale response

    Returns:
        (np.ndarray): Response that will decrease abs(to_xyz - from_xyz)

    """
    # move down if only z diff
    if np.linalg.norm(to_xyz[:2] - from_xyz[:2]) < 0.02:
        # print("move z down")
        to_xyz = to_xyz
    # move up if z < 0.3
    elif from_xyz[2] < 0.3:
        # print("move z up")
        to_xyz = from_xyz * np.array([1., 1., 0.]) + np.array([0., 0., 0.3])
    # move y- if x diff and y > 0.3
    elif abs(to_xyz[0] - from_xyz[0]) > 0.01 and from_xyz[1] > 0.5:
        # print("move y near")
        to_xyz = from_xyz * np.array([1., 0., 1.]) + np.array([0., 0.45, 0.])
    # move x if x diff and y < 0.3
    elif abs(to_xyz[0] - from_xyz[0]) > 0.01:
        # print("move x")
        to_xyz = from_xyz * np.array([0., 1., 1.]) + to_xyz * np.array([1., 0., 0.])
    # move y+ if x not diff
    else:
        # print("move y far")
        to_xyz = from_xyz * np.array([0., 0., 1.]) + to_xyz * np.array([1., 1., 0.])
    error = to_xyz - from_xyz
    response = p * error

    if np.any(np.absolute(response) > 1.):
        warnings.warn('Constant(s) may be too high. Environments clip response to [-1, 1]')

    return response
