from typing import Callable
from copy import deepcopy
from dataclasses import dataclass
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


@dataclass(frozen=True)
class States:
    CUP_STATE_AIR = 'AIR'
    CUP_STATE_DRAWER = 'DRAWER'
    CUP_STATE_DESK = 'DESK'
    CUP_STATE_MACHINE = 'MACHINE'
    CUP_STATE_SHELF = 'SHELF'
    DRAWER_STATE_OPENED = 'OPENED'
    DRAWER_STATE_CLOSED = 'CLOSED'

STATES = States()


PRECONDITIONS = {
    'coffee-button': {'cup': lambda x: x == STATES.CUP_STATE_MACHINE},
    'coffee-pull': {'cup': lambda x: x == STATES.CUP_STATE_MACHINE},
    'coffee-push': {'cup': lambda x: x == STATES.CUP_STATE_AIR},
    'drawer-close': {'cup': lambda x: x != STATES.CUP_STATE_AIR,
                     'drawer': lambda x: x == STATES.DRAWER_STATE_OPENED},
    'drawer-open': {'cup': lambda x: x != STATES.CUP_STATE_AIR,
                    'drawer': lambda x: x == STATES.DRAWER_STATE_CLOSED},
    'drawer-pick': {'cup': lambda x: x == STATES.CUP_STATE_DRAWER,
                    'drawer': lambda x: x == STATES.DRAWER_STATE_OPENED},
    'drawer-place': {'cup': lambda x: x == STATES.CUP_STATE_AIR,
                    'drawer': lambda x: x == STATES.DRAWER_STATE_OPENED},
    'desk-pick': {'cup': lambda x: x == STATES.CUP_STATE_DESK},
    'desk-place': {'cup': lambda x: x == STATES.CUP_STATE_AIR},
    'reset': {'cup': lambda x: x != STATES.CUP_STATE_AIR},
}
POSTSTATES = {
    'coffee-button': {},
    'coffee-pull': {'cup': STATES.CUP_STATE_AIR},
    'coffee-push': {'cup': STATES.CUP_STATE_MACHINE},
    'drawer-close': {'drawer': STATES.DRAWER_STATE_CLOSED},
    'drawer-open': {'drawer': STATES.DRAWER_STATE_OPENED},
    'drawer-pick': {'cup': STATES.CUP_STATE_AIR},
    'drawer-place': {'cup': STATES.CUP_STATE_DRAWER},
    'desk-pick': {'cup': STATES.CUP_STATE_AIR},
    'desk-place': {'cup': STATES.CUP_STATE_DESK},
    'reset': {},
}


def check_task_cond(task: str, states: dict[str, str]) -> bool:
    obj2conds: dict[str, Callable] = PRECONDITIONS[task]
    success = True
    for obj, cond in obj2conds.items():
        success: bool = success and cond(states[obj])
        if not success:
            break
    return success


def change_state(task: str, states: dict[str, str]) -> dict[str, str]:
    states = deepcopy(states)
    obj2state: dict[str, str] = POSTSTATES[task]
    for obj, new_state in obj2state.items():
        states[obj] = new_state
    return states


def check_if_state_valid(states: dict[str, str]):
    assert 'cup' in states.keys()
    assert 'drawer' in states.keys()
    VALID_CUP_STATES = (STATES.CUP_STATE_AIR, STATES.CUP_STATE_DRAWER,
                        STATES.CUP_STATE_DESK, STATES.CUP_STATE_MACHINE)
    VALID_DRAWER_STATES = (STATES.DRAWER_STATE_CLOSED,
                           STATES.DRAWER_STATE_OPENED)
    assert states['cup'] in VALID_CUP_STATES
    assert states['drawer'] in VALID_DRAWER_STATES


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
