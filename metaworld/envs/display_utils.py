from typing import Callable
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import matplotlib.colors
from metaworld.envs.env_utils import get_logger

import warnings


logger = get_logger(__name__)

# COLOR_LIST = [
#     "#F27970",
#     "#BB9727",
#     "#54B345",
#     "#32B897",
#     "#05B9E2",
#     "#8983BF",
#     "#C76DA2",
# ]
COLOR_LIST = [
    # 标准颜色
    ("black",   "#000000"),
    ("white",   "#FFFFFF"),
    ("red",     "#FF0000"),
    ("green",   "#00FF00"),
    ("blue",    "#0000FF"),
    ("yellow",  "#FFFF00"),
    ("cyan",    "#00FFFF"),
    ("carmine", "#FF00FF"),
    ("gray",    "#808080"),
    ("orange",  "#FFA500"),
    ("purple",  "#800080"),
    ("brown",   "#A52A2A"),
]
RGB_COLOR_LIST = [(color[0], matplotlib.colors.to_rgb(color[1])) for color in COLOR_LIST]

QUAT_LIST = [
    [1., 0., 0., 0.],                               # 正常状态
    [np.sin(np.pi/4), 0., 0., np.sin(np.pi/4)],     # 俯视顺时针旋转90
    [np.sin(np.pi/4), 0., 0., -np.sin(np.pi/4)],    # 俯视逆时针旋转90
    [0., 0., 0., 1.],                               # 俯视旋转180
]


@dataclass(frozen=True)
class Tasks:
    COFFEE_BUTTON = 'coffee-button'
    COFFEE_PULL = 'coffee-pull'
    COFFEE_PUSH = 'coffee-push'
    DRAWER_CLOSE = 'drawer-close'
    DRAWER_OPEN = 'drawer-open'
    DRAWER_PICK = 'drawer-pick'
    DRAWER_PLACE = 'drawer-place'
    DESK_PICK = 'desk-pick'
    DESK_PLACE = 'desk-place'
    RESET_HAND = 'reset-hand'
    BIN_PICK = 'bin-pick'
    BIN_PLACE = 'bin-place'


@dataclass(frozen=True)
class StateKeys:
    CUP = 'cup'
    DRAWER = 'drawer'
    COFFEE_MACHINE = 'coffee_machine'
    HANDLE_OBJECT = 'handle_object'
    CUP_IN_DRAWER = 'cup_in_drawer'
    DRAWER_CONTAINS_CUP = 'drawer_contains_cup'
    GRIPPER = 'gripper'


@dataclass(frozen=True)
class States:
    CUP_STATE_AIR = 'AIR'
    CUP_STATE_DRAWER = 'DRAWER'
    CUP_STATE_DESK = 'DESK'
    CUP_STATE_BIN = 'BIN'
    CUP_STATE_MACHINE = 'MACHINE'
    CUP_STATE_SHELF = 'SHELF'
    CUP_OUTSIDE_DRAWER = None
    DRAWER_STATE_OPENED = 'OPENED'
    DRAWER_STATE_CLOSED = 'CLOSED'
    DRAWER_IS_EMPTY = None
    COFFEE_MACHINE_IS_EMPTY = None
    GRIPPER_STATE_AIR = 'AIR'
    GRIPPER_STATE_CUP = 'CUP'


TASKS = Tasks()
STATES = States()
STATE_KEYS = StateKeys()


PRECONDITIONS = {
    TASKS.COFFEE_BUTTON: {
        STATE_KEYS.CUP:
            lambda x: x == STATES.CUP_STATE_MACHINE,
        STATE_KEYS.GRIPPER:
            lambda x: x == STATES.GRIPPER_STATE_AIR,
        },
    TASKS.COFFEE_PULL: {
        STATE_KEYS.CUP:
            lambda x: x == STATES.CUP_STATE_MACHINE,
        STATE_KEYS.COFFEE_MACHINE:
            lambda x: x != STATES.COFFEE_MACHINE_IS_EMPTY,
        STATE_KEYS.GRIPPER:
            lambda x: x == STATES.GRIPPER_STATE_AIR,
        },
    TASKS.COFFEE_PUSH: {
        STATE_KEYS.CUP:
            lambda x: x == STATES.CUP_STATE_AIR,
        STATE_KEYS.COFFEE_MACHINE:
            lambda x: x == STATES.COFFEE_MACHINE_IS_EMPTY,
        STATE_KEYS.GRIPPER:
            lambda x: x == STATES.GRIPPER_STATE_CUP,
        },
    TASKS.DRAWER_CLOSE: {
        STATE_KEYS.CUP:
            lambda x: x != STATES.CUP_STATE_AIR,
        STATE_KEYS.DRAWER:
            lambda x: x == STATES.DRAWER_STATE_OPENED,
        STATE_KEYS.GRIPPER:
            lambda x: x == STATES.GRIPPER_STATE_AIR,
        },
    TASKS.DRAWER_OPEN: {
        STATE_KEYS.CUP:
            lambda x: x != STATES.CUP_STATE_AIR,
        STATE_KEYS.DRAWER:
            lambda x: x == STATES.DRAWER_STATE_CLOSED,
        STATE_KEYS.GRIPPER:
            lambda x: x == STATES.GRIPPER_STATE_AIR,
        },
    TASKS.DRAWER_PICK: {
        STATE_KEYS.CUP:
            lambda x: x == STATES.CUP_STATE_DRAWER,
        STATE_KEYS.DRAWER:
            lambda x: x == STATES.DRAWER_STATE_OPENED,
        STATE_KEYS.CUP_IN_DRAWER:
            lambda x: x is not STATES.CUP_OUTSIDE_DRAWER,
        STATE_KEYS.DRAWER_CONTAINS_CUP:
            lambda x: x is not STATES.DRAWER_IS_EMPTY,
        STATE_KEYS.GRIPPER:
            lambda x: x == STATES.GRIPPER_STATE_AIR,
        },
    TASKS.DRAWER_PLACE: {
        STATE_KEYS.CUP:
            lambda x: x == STATES.CUP_STATE_AIR,
        STATE_KEYS.DRAWER:
            lambda x: x == STATES.DRAWER_STATE_OPENED,
        STATE_KEYS.CUP_IN_DRAWER:
            lambda x: x is STATES.CUP_OUTSIDE_DRAWER,
        STATE_KEYS.DRAWER_CONTAINS_CUP:
            lambda x: x is STATES.DRAWER_IS_EMPTY,
        STATE_KEYS.GRIPPER:
            lambda x: x == STATES.GRIPPER_STATE_CUP,
        },
    TASKS.DESK_PICK: {
        STATE_KEYS.CUP:
            lambda x: x == STATES.CUP_STATE_DESK,
        STATE_KEYS.GRIPPER:
            lambda x: x == STATES.GRIPPER_STATE_AIR,
        },
    TASKS.DESK_PLACE: {
        STATE_KEYS.CUP:
            lambda x: x == STATES.CUP_STATE_AIR,
        STATE_KEYS.GRIPPER:
            lambda x: x == STATES.GRIPPER_STATE_CUP,
        },
    TASKS.BIN_PICK: {
        STATE_KEYS.CUP:
            lambda x: x == STATES.CUP_STATE_BIN
        },
    TASKS.BIN_PLACE: {
        STATE_KEYS.CUP:
            lambda x: x == STATES.CUP_STATE_AIR
        },
    TASKS.RESET_HAND: {
        STATE_KEYS.CUP:
            lambda x: x != STATES.CUP_STATE_AIR,
        STATE_KEYS.GRIPPER:
            lambda x: x == STATES.GRIPPER_STATE_AIR,
        },
}
POSTSTATES = {
    TASKS.COFFEE_BUTTON: {},
    TASKS.COFFEE_PULL: {
        STATE_KEYS.CUP:
            lambda x: STATES.CUP_STATE_AIR,
        STATE_KEYS.COFFEE_MACHINE:
            lambda x: None,
        STATE_KEYS.GRIPPER:
            lambda x: STATES.GRIPPER_STATE_CUP,
        },
    TASKS.COFFEE_PUSH: {
        STATE_KEYS.CUP:
            lambda x: STATES.CUP_STATE_MACHINE,
        STATE_KEYS.COFFEE_MACHINE:
            lambda x: x[STATE_KEYS.HANDLE_OBJECT][STATE_KEYS.CUP],
        STATE_KEYS.GRIPPER:
            lambda x: STATES.GRIPPER_STATE_AIR,
        },
    TASKS.DRAWER_CLOSE: {
        STATE_KEYS.DRAWER:
            lambda x: STATES.DRAWER_STATE_CLOSED,
        STATE_KEYS.GRIPPER:
            lambda x: STATES.GRIPPER_STATE_AIR,
        },
    TASKS.DRAWER_OPEN: {
        STATE_KEYS.DRAWER:
            lambda x: STATES.DRAWER_STATE_OPENED,
        STATE_KEYS.GRIPPER:
            lambda x: STATES.GRIPPER_STATE_AIR,
        },
    TASKS.DRAWER_PICK: {
        STATE_KEYS.CUP:
            lambda x: STATES.CUP_STATE_AIR,
        STATE_KEYS.CUP_IN_DRAWER:
            lambda x: None,
        STATE_KEYS.DRAWER_CONTAINS_CUP:
            lambda x: None,
        STATE_KEYS.GRIPPER:
            lambda x: STATES.GRIPPER_STATE_CUP,
        },
    TASKS.DRAWER_PLACE: {
        STATE_KEYS.CUP:
            lambda x: STATES.CUP_STATE_DRAWER,
        STATE_KEYS.CUP_IN_DRAWER:
            lambda x: x[STATE_KEYS.HANDLE_OBJECT][STATE_KEYS.DRAWER],
        STATE_KEYS.DRAWER_CONTAINS_CUP:
            lambda x: x[STATE_KEYS.HANDLE_OBJECT][STATE_KEYS.CUP],
        STATE_KEYS.GRIPPER:
            lambda x: STATES.GRIPPER_STATE_AIR,
        },
    TASKS.DESK_PICK: {
        STATE_KEYS.CUP:
            lambda x: STATES.CUP_STATE_AIR,
        STATE_KEYS.GRIPPER:
            lambda x: STATES.GRIPPER_STATE_CUP,
        },
    TASKS.DESK_PLACE: {
        STATE_KEYS.CUP:
            lambda x: STATES.CUP_STATE_DESK,
        STATE_KEYS.GRIPPER:
            lambda x: STATES.GRIPPER_STATE_AIR,
        },
    TASKS.BIN_PICK: {
        STATE_KEYS.CUP:
            lambda x: STATES.CUP_STATE_AIR,
        },
    TASKS.BIN_PLACE: {
        STATE_KEYS.CUP:
            lambda x: STATES.CUP_STATE_BIN
        },
    TASKS.RESET_HAND: {},
}
TASK_RANDOM_PROBABILITY = {
    # TASKS.COFFEE_BUTTON: 1/0.052,
    # TASKS.COFFEE_PULL: 1/0.052,
    # TASKS.COFFEE_PUSH: 1/0.065,
    # TASKS.DRAWER_CLOSE: 1/0.111,
    # TASKS.DRAWER_OPEN: 1/0.137,
    # TASKS.DRAWER_PICK: 1/0.019,
    # TASKS.DRAWER_PLACE: 1/0.026,
    # TASKS.DESK_PICK: 1/0.105,
    # TASKS.DESK_PLACE: 1/0.065,
    # TASKS.BIN_PICK: 1/0.055,
    # TASKS.BIN_PLACE: 1/0.065,
    # TASKS.RESET_HAND: 1/0.248,
    TASKS.COFFEE_BUTTON: 1,
    TASKS.COFFEE_PULL: 1,
    TASKS.COFFEE_PUSH: 1,
    TASKS.DRAWER_CLOSE: 1,
    TASKS.DRAWER_OPEN: 1,
    TASKS.DRAWER_PICK: 1,
    TASKS.DRAWER_PLACE: 1,
    TASKS.DESK_PICK: 1,
    TASKS.DESK_PLACE: 1,
    TASKS.BIN_PICK: 1,
    TASKS.BIN_PLACE: 1,
    TASKS.RESET_HAND: 0.05,
}


def check_task_cond(task: str, states: dict[str, str]) -> bool:
    states = deepcopy(states)
    obj2conds: dict[str, Callable] = PRECONDITIONS[task]
    success = True
    for obj, cond in obj2conds.items():
        if obj not in states.keys():
            # display.py does not contain state `drawer_contains_cup`
            continue
        success: bool = success and cond(states[obj])
        if not success:
            break
    return success


def change_state(task: str, states: dict[str, str]) -> dict[str, str]:
    states = deepcopy(states)
    obj2state: dict[str, str] = POSTSTATES[task]
    for obj, new_state in obj2state.items():
        if obj in states.keys():
            states[obj] = new_state(states)
    return states


def parse_task(task: str) -> str:
    return task.split(")")[-1]


def check_if_state_valid(states: dict[str, str]):
    assert 'cup' in states.keys()
    assert 'drawer' in states.keys()
    VALID_CUP_STATES = (STATES.CUP_STATE_AIR, STATES.CUP_STATE_DRAWER,
                        STATES.CUP_STATE_DESK, STATES.CUP_STATE_MACHINE)
    VALID_DRAWER_STATES = (STATES.DRAWER_STATE_CLOSED,
                           STATES.DRAWER_STATE_OPENED)
    assert states['cup'] in VALID_CUP_STATES
    assert states['drawer'] in VALID_DRAWER_STATES


def detect_insert_missing_tasks(tasklist: list[str]) -> list[str]:
    tasklist = deepcopy(tasklist)
    while True:
        fixed_list = detect_insert_missing_task(tasklist)
        if len(fixed_list) <= len(tasklist):
            # Some missing tasks cannot be correctly found or the detection
            # and fixing progress is done.
            return fixed_list
        tasklist = fixed_list


def detect_insert_missing_task(tasklist: list[str]) -> list[str]:
    states = dict(cup=STATES.CUP_STATE_DESK,
                  drawer=STATES.DRAWER_STATE_CLOSED)
    tasklist_fixed = deepcopy(tasklist)
    for idx, task in enumerate(tasklist):
        if check_task_cond(task, states):
            states = change_state(task, states)
        else:
            # Current task cannot be executed, at least one task missing.
            missing_task = find_missing_task(task, states)
            if missing_task is None:
                message = ('Some tasks are missing but cannot be '
                           'automatically detected. The following tasks '
                           f'will be discarded: {tasklist[idx:]}')
                # warnings.warn(message)
                logger.warn(message)
                return tasklist[:idx]
            else:
                message = (f'Find a potential missing task {missing_task}.')
                # warnings.warn(message)
                logger.warn(message)
                tasklist_fixed.insert(idx, missing_task)
                return tasklist_fixed
    return tasklist_fixed


def find_missing_task(task: str,
                      states: dict[str, str]) -> str:
    for cand_task in find_potential_tasks(states):
        simulated_states = change_state(cand_task, deepcopy(states))
        if check_task_cond(task, simulated_states):
            return cand_task
    return None


def find_potential_tasks(states: dict[str, str]) -> list[str]:
    potential_tasks = []
    for cand_task in POSTSTATES.keys():
        if check_task_cond(cand_task, states):
            potential_tasks.append(cand_task)
    return potential_tasks


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
