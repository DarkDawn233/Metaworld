
import warnings
from typing import Dict
from copy import deepcopy
import logging
from dataclasses import dataclass
import numpy as np
from gym.spaces import Box
import random

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_display_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import _assert_task_is_set
from metaworld.envs.env_utils import get_logger

from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_base import SawyerXYZEnvDisplay
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_coffee_button_display import SawyerCoffeeButtonEnvV2Display
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_coffee_pull_display import SawyerCoffeePullEnvV2Display
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_coffee_push_display import SawyerCoffeePushEnvV2Display
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_drawer_close_display import SawyerDrawerCloseEnvV2Display
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_drawer_open_display import SawyerDrawerOpenEnvV2Display
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_drawer_place_display import SawyerDrawerPlaceEnvV2Display
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_drawer_pick_display import SawyerDrawerPickEnvV2Display
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_desk_pick_display import SawyerDeskPickEnvV2Display
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_desk_place_display import SawyerDeskPlaceEnvV2Display
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_bin_pick_display import SawyerBinPickEnvV2Display
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_bin_place_display import SawyerBinPlaceEnvV2Display
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_reset_display import SawyerResetHandEnvV2Display

from metaworld.envs.display_utils import (random_grid_pos,
                                          check_task_cond,
                                          change_state,
                                          parse_task,
                                          find_missing_task,
                                          TASKS,
                                          STATES,
                                          RGB_COLOR_LIST,
                                          STATE_KEYS,
                                          TASK_RANDOM_PROBABILITY)


NAME2ENVS: Dict[str, SawyerXYZEnvDisplay] = {
    TASKS.COFFEE_BUTTON: SawyerCoffeeButtonEnvV2Display,
    TASKS.COFFEE_PULL: SawyerCoffeePullEnvV2Display,
    TASKS.COFFEE_PUSH: SawyerCoffeePushEnvV2Display,
    TASKS.DRAWER_CLOSE: SawyerDrawerCloseEnvV2Display,
    TASKS.DRAWER_OPEN: SawyerDrawerOpenEnvV2Display,
    TASKS.DRAWER_PICK: SawyerDrawerPickEnvV2Display,
    TASKS.DRAWER_PLACE: SawyerDrawerPlaceEnvV2Display,
    TASKS.DESK_PICK: SawyerDeskPickEnvV2Display,
    TASKS.DESK_PLACE: SawyerDeskPlaceEnvV2Display,
    TASKS.RESET_HAND: SawyerResetHandEnvV2Display,
}


logger = get_logger(__name__)


class SawyerEnvV2Display3D3M(
            SawyerCoffeeButtonEnvV2Display,
            SawyerCoffeePullEnvV2Display,
            SawyerCoffeePushEnvV2Display,
            SawyerDrawerCloseEnvV2Display,
            SawyerDrawerOpenEnvV2Display,
            SawyerDrawerPickEnvV2Display,
            SawyerDrawerPlaceEnvV2Display,
            SawyerDeskPickEnvV2Display,
            SawyerDeskPlaceEnvV2Display,
            SawyerResetHandEnvV2Display,
        ):

    TASK_LIST = list(NAME2ENVS.keys())
    max_path_length = 1e8

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.0)
        obj_high = (0.1, 0.9, 0.0)

        super(SawyerXYZEnvDisplay, self).__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0.4, 0.85, 0], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

        self.task_list = []
        self.now_task = ""
        self.task_step = 0
        self.task_done = True
        self._cup_machine_offset = 0.28

        self.random_generate_task = False
        self._states = {
            STATE_KEYS.CUP: {idx: None for idx in range(3)},
            STATE_KEYS.DRAWER: {idx: None for idx in range(3)},
            # TODO Adaptively set the number of coffee machine.
            STATE_KEYS.COFFEE_MACHINE: {0: None},
            STATE_KEYS.CUP_IN_DRAWER: {idx: None for idx in range(3)},
            STATE_KEYS.DRAWER_CONTAINS_CUP: {idx: None for idx in range(3)},
            STATE_KEYS.HANDLE_OBJECT: {STATE_KEYS.CUP: 0,
                                       STATE_KEYS.DRAWER: 0,
                                       STATE_KEYS.COFFEE_MACHINE: 0}}

        self.fix_reset_flag = False # 固定重置环境

        self.last_step_mug_pos = None

        # self.target_mug_id = 0
        # self.target_drawer_id = 0
        # self.target_coffee_machine_id = 0

        self.color2item_dict = {}

    @property
    def model_name(self):
        return full_display_path_for('sawyer_xyz/_sawyer_display_3d3m.xml')
    
    def _random_init_drawer_pos(self):
        x_list = [-0.4, -0.12, 0.4]
        self.drawer_init_pos = []
        for i in range(3):
            pos = [x_list[i], 0.85, 0]
            pos = np.array(pos)
            self.sim.model.body_pos[
                self.sim.model.body_name2id('drawer'+str(i))
            ] = pos
            self.drawer_init_pos.append(pos)

            quat = np.array([1., 0., 0., 0.])
            self.sim.model.body_quat[
                self.sim.model.body_name2id('drawer'+str(i))
            ] = quat
    
    def _random_coffee_machine_init_pos(self):
        pos = [0.12, 0.85, 0]
        pos = np.array(pos)
        self.sim.model.body_pos[
            self.sim.model.body_name2id('coffee_machine')
        ] = pos
        self.coffee_machine_init_pos = pos

        quat = np.array([1., 0., 0., 0.])
        self.sim.model.body_quat[
            self.sim.model.body_name2id('coffee_machine')
        ] = quat

    
    def _random_init_mug_pos(self):
        forbid_list = [((drawer_pos[0]-0.25, drawer_pos[0]+0.25),
                        (drawer_pos[1]-0.4, drawer_pos[1]+0.2)) for drawer_pos in self.drawer_init_pos]
        forbid_list.append(
            ((self.coffee_machine_init_pos[0]-0.25, self.coffee_machine_init_pos[0]+0.25),
             (self.coffee_machine_init_pos[1]-0.3, self.coffee_machine_init_pos[1]+0.2))
        )
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        self.mug_init_pos = []
        for i in range(3):
            x_range = [-0.5, 0.5]
            y_range = [0.4, 0.9]
            x, y = random_grid_pos(x_range, y_range, forbid_list)
            pos = [x, y, 0]
            forbid_list.append(((x-0.1, x+0.1), (y-0.1, y+0.1)))
            pos_quat = np.array(pos + [1., 0., 0., 0.])
            self.mug_init_pos.append(pos)
            addr_mug = self.model.get_joint_qpos_addr('mug_obj'+str(i))
            qpos[addr_mug[0]: addr_mug[1]] = pos_quat

        self.set_state(qpos, qvel)
    
    def _reset_hand(self, steps=50):
        # Overwrite `_reset_hand` method. Initial hand pos is [0.0, 0.4, 0.2].
        # lower_bound = (-0.5, 0.40, 0.05)
        # upper_bound = ( 0.5, 0.80, 0.45)
        # self.hand_init_pos = np.random.uniform(lower_bound, upper_bound)
        # self.hand_init_pos = upper_bound
        mocap_quat = np.array([1, 0, 1, 0])
        for _ in range(steps):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', mocap_quat)
            self.do_simulation([-1, 1], self.frame_skip)
        self.init_tcp = self.tcp_center
        # print(f'Hand position randomly set as {self.hand_init_pos}.')

    def _random_init_hand_pos(self, pos=None):
        self.hand_init_pos = pos
        self._reset_hand()
    
    def _random_init_color(self):

        def set_model_rgba(name: str, rgb):
            if "drawer" in name:
                model_name = [name.replace('drawer', 'drawercase_tmp'), name.replace('drawer', 'drawer_tmp')]
            elif "mug" in name:
                model_name = [name, name.replace('mug', 'handle')]
            elif name == 'coffee_machine_body':
                model_name = ['coffee_machine_body1', 'coffee_machine_body2']
            else:
                model_name = [name]

            rgba = list(rgb) + [1.]
            for name in model_name:
                self.sim.model.geom_rgba[
                    self.sim.model.geom_name2id(name)] = rgba

        # model_name_list = ['drawer', 'coffee_machine_body', 'shelf', 'mug']
        model_name_list = ['drawer0', 'drawer1', 'drawer2'] + ['coffee_machine_body'] + ['mug0', 'mug1', 'mug2']
        rgb_list = random.sample(RGB_COLOR_LIST, len(model_name_list))
        self.color2item_dict = {}
        for model_name, rgb in zip(model_name_list, rgb_list):
            set_model_rgba(model_name, rgb[1])
            self.color2item_dict[rgb[0]] = model_name

    def reset_model(self):
        """
        random_level:   0 - 固定物体位置
                        1 - 不改变drawer/coffee_machine/shelf朝向 随机三个物体的左中右 
        """

        self.task_list = []
        self.now_task = ""
        self.task_step = 0
        self.after_success_cnt = 0
        self.task_done = True

        self.last_step_mug_pos = None

        self._random_table_and_floor()
            # self.random_obj_list = ['drawer', 'coffee_machine', 'shelf']

        # self._random_drawer_init(self.fix_reset_flag)
        # self._random_coffee_machine_init(self.fix_reset_flag)
        # self._random_shelf_init()
        # self._random_bin_init(self.fix_reset_flag)
        # self._random_init_mug(self.fix_reset_flag)
        self._random_init_drawer_pos()
        self._random_coffee_machine_init_pos()
        self._random_init_mug_pos()

        self._random_init_color()
            
        self._random_init_hand_pos([0, 0.4, 0.4])

        self._target_pos = np.zeros(3)
        
        self.obj_init_pos = self._get_pos_objects()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        for idx in self._states[STATE_KEYS.DRAWER].keys():
            self._states[STATE_KEYS.DRAWER][idx] = STATES.DRAWER_STATE_CLOSED
        for idx in self._states[STATE_KEYS.CUP].keys():
            self._states[STATE_KEYS.CUP][idx] = STATES.CUP_STATE_DESK
        for idx in self._states[STATE_KEYS.COFFEE_MACHINE].keys():
            self._states[STATE_KEYS.COFFEE_MACHINE][idx] = \
                STATES.COFFEE_MACHINE_IS_EMPTY
        logger.info(f"self.obj_init_pos: {self.obj_init_pos}")
        return self._get_obs()

    def fix_reset(self, flag=True):
        self.fix_reset_flag = flag

    def _get_pos_objects(self):
        if not hasattr(self, 'task_list') or len(self.task_list) == 0 or self.now_task == "":
        # if self.now_task == "":
            return np.zeros(3)
        # now_task = self.task_list[0]
        if self.now_task == TASKS.COFFEE_PULL:
            results = self.get_body_com('obj'+str(self.target_mug_id))
        elif self.now_task == TASKS.COFFEE_PUSH:
            results = self.get_body_com('obj'+str(self.target_mug_id))
        elif self.now_task == TASKS.DRAWER_CLOSE:
            results = self.get_body_com('drawer_link'+str(self.target_drawer_id)) + np.array([.0, -.16, .05])
        elif self.now_task == TASKS.DRAWER_OPEN:
            results = self.get_body_com('drawer_link'+str(self.target_drawer_id)) + np.array([.0, -.16, .05])
        elif self.now_task == TASKS.DRAWER_PICK:
            results = np.hstack((
                self.get_body_com('obj'+str(self.target_mug_id)),
                self.get_body_com('drawer_link'+str(self.target_drawer_id))
            ))
        elif self.now_task == TASKS.DRAWER_PLACE:
            results = np.hstack((
                self.get_body_com('obj'+str(self.target_mug_id)),
                self.get_body_com('drawer_link'+str(self.target_drawer_id))
            ))
        elif self.now_task == TASKS.DESK_PICK:
            results = self.get_body_com('obj'+str(self.target_mug_id))
        elif self.now_task == TASKS.DESK_PLACE:
            results = self.get_body_com('obj'+str(self.target_mug_id))
        elif self.now_task in NAME2ENVS.keys():
            results = NAME2ENVS[self.now_task]._get_pos_objects(self)
        else:
            raise NotImplementedError()
        return results

    def _get_quat_objects(self):
        # if not hasattr(self, 'task_list') or len(self.task_list) == 0 or self.task_list[0] is None:
        if not hasattr(self, 'task_list') or len(self.task_list) == 0 or self.now_task == "":
        # if self.now_task == "":
            return np.zeros(4)
        # now_task = self.task_list[0]
        if self.now_task == TASKS.COFFEE_BUTTON:
            results = np.array([1., 0., 0., 0.])
        elif self.now_task == TASKS.COFFEE_PULL:
            results = self.get_body_quat('obj'+str(self.target_mug_id))
        elif self.now_task == TASKS.COFFEE_PUSH:
            results = self.get_body_quat('obj'+str(self.target_mug_id))
        elif self.now_task == TASKS.DRAWER_CLOSE:
            results = self.get_body_quat('drawer_link'+str(self.target_drawer_id))
        elif self.now_task == TASKS.DRAWER_OPEN:
            results = self.get_body_quat('drawer_link'+str(self.target_drawer_id))
        elif self.now_task == TASKS.DRAWER_PICK:
            results = np.hstack((
                self.get_body_quat('mug'+str(self.target_mug_id)),
                self.get_body_quat('drawer_link'+str(self.target_drawer_id))
            ))
        elif self.now_task == TASKS.DRAWER_PLACE:
            results = np.hstack((
                self.get_body_quat('mug'+str(self.target_mug_id)),
                self.get_body_quat('drawer_link'+str(self.target_drawer_id))
            ))
        elif self.now_task == TASKS.DESK_PICK:
            results = self.get_body_quat('obj'+str(self.target_mug_id))
        elif self.now_task == TASKS.DESK_PLACE:
            results = self.get_body_quat('obj'+str(self.target_mug_id))
        elif self.now_task in NAME2ENVS.keys():
            results = NAME2ENVS[self.now_task]._get_quat_objects(self)
        else:
            raise NotImplementedError()
        return results
    
    def _check_task_list(self, task_list):
        """
        要求所有任务名称包含括号 "()"
        """
        # TODO  区分 原task_list 和 解析后 task list
        for task in task_list:
            if parse_task(task) not in self.TASK_LIST:
                raise ValueError(f"Task name error: no task named {task}")
        
    def reset_task_list(self, task_list):
        self._check_task_list(task_list)
        self.task_list = task_list
        self.task_step = 0
        self.task_done = False
    
    def append_task_list(self, task_list):
        self._check_task_list(task_list)
        self.task_list += task_list
    
    def _reset_button_offsets(self, pos=None):
        pos = 0 if pos is None else pos
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        addr = self.model.get_joint_qpos_addr('goal_slidey')
        qpos[addr] = pos
        self.set_state(qpos, qvel)
    
    def _judge_grab(self, id):
        pos_mug = self.get_body_com('obj'+id)
        for j in range(3):
            pos_drawer = self.get_body_com('drawer_link'+str(j))
            if np.linalg.norm(pos_drawer[:2] - pos_mug[:2]) <= 0.1:
                return pos_mug[2] > 0.06
        return pos_mug[2] > 0.0
    
    def _reset_state(self, success):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        # now_task = "" if len(self.task_list) == 0 else self.task_list[0]
        for i in range(3):
            id = str(i)
            
            # mug_i
            if not self._judge_grab(id):
                addr_mug = self.model.get_joint_qpos_addr('mug_obj'+id)
                if "place" in self.now_task and self.target_mug_id == i:
                    qpos[addr_mug[0]+0: addr_mug[0]+3] = self._target_pos
                qpos[addr_mug[0]+3: addr_mug[0]+7] = np.array(np.array([1., 0., 0., 0.]))

            addr_drawer = self.model.get_joint_qpos_addr('drawer_goal_slidey'+id)

            if self.now_task not in ["drawer-close", "drawer-open"] or self.target_drawer_id != i:
                if self._states[STATE_KEYS.DRAWER][i] == STATES.DRAWER_STATE_CLOSED:
                    qpos[addr_drawer] = 0
                else:
                    qpos[addr_drawer] = -0.16
            elif self.now_task == "drawer-close" and success and self.target_drawer_id == i:
                qpos[addr_drawer] = 0
            elif self.now_task == "drawer-open" and success and self.target_drawer_id == i:
                qpos[addr_drawer] = -0.16

        self.set_state(qpos, qvel)
    
    @_assert_task_is_set
    def step(self, action):
        if self.task_step == 0 and len(self.task_list) > 0:
            # 分解当前任务 -> target_mug_id / target_drawer_id
            self._task_map()
        return super(SawyerEnvV2Display3D3M, self).step(action)


    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        if len(self.task_list) == 0:
            self.now_task = ""
            self.task_step = 0
            info = {
                'success': float(False),
                'task_name': None,
                'task_step': 0,
            }
            self._reset_state(info['success'])
            return 0., info

        if self.task_step == 0:
            if not check_task_cond(self.now_task, self.target_states):
                message = (f'Task {self.now_task} is invalid for state: '
                           f'{self.states}.')
                logger.warn(message)
                # warnings.warn(message)
                # TODO 下一步工作：如何补全miss task
                missing_task = find_missing_task(self.now_task, self.target_states)
                if missing_task is None:
                    message = (f'Cannot find the missing task, stopped.')
                    logger.warn(message)
                    # warnings.warn(message)
                    self.task_step = 0
                    info = {
                        'success': float(False),
                        'task_name': None,
                        'task_step': 0,
                    }
                    self._reset_state(info['success'])
                    self.task_list = list()
                    return 0., info
                else:
                    message = (f'Find a potential missing task '
                               f'{missing_task}, execute it first.')
                    logger.warn(message)
                    # warnings.warn(message)
                    self.task_list.insert(0, missing_task)
                    self.now_task = missing_task
            self._reset_button_offsets()
            logger.info(f'TaskList: {self.task_list}')
            logger.info(f'CURRENT TASK: {self.now_task}')

        if self.now_task == TASKS.COFFEE_BUTTON:
            if self.task_step == 0:
                self.max_dist = 0.09
                pos_button = self.coffee_machine_init_pos + np.array([.0, -self._cup_machine_offset, .3])
                self._target_pos = pos_button + np.array([.0, self.max_dist, .0])
                self.quat = np.array([1., 0., 0., 0.])
                self.succeed = False
        elif self.now_task == TASKS.COFFEE_PULL:
            if self.task_step == 0:
                self.max_dist = 0.03
                # assert hasattr(self, 'mug_init_pos')
                self._target_pos = self._get_mug_pick_pos(self.target_mug_id)
                self.quat = np.array([1., 0., 0., 0.])
                self.succeed = False
        elif self.now_task == TASKS.COFFEE_PUSH:
            if self.task_step == 0:
                self.max_dist = 0.03
                pos_goal = self.coffee_machine_init_pos + np.array([.0, -self._cup_machine_offset, .0])
                self._target_pos = pos_goal
                self.quat = np.array([1., 0., 0., 0.])
                self.succeed = False
        elif self.now_task == TASKS.DRAWER_CLOSE:
            if self.task_step == 0:
                self._target_pos = self.get_body_com('drawer'+str(self.target_drawer_id)) + np.array([.0, -.16, .09])
                self.obj_init_pos = self._get_pos_objects()
        elif self.now_task == TASKS.DRAWER_OPEN:
            if self.task_step == 0:
                self.maxDist = 0.16
                self._target_pos = self.get_body_com('drawer'+str(self.target_drawer_id)) + np.array([.0, -.16 - self.maxDist, .09])
                self._handle_pos_init = self._target_pos + np.array([.0, self.maxDist, .0])
                self.quat_index = 0
                self.drawer_link_name = 'drawer_link' + str(self.target_drawer_id)
        elif self.now_task == TASKS.DRAWER_PICK:
            if self.task_step == 0:
                self._target_pos = self.get_body_com('obj'+str(self.target_mug_id)) + np.array([.0, .0, .3])
                self.obj_init_pos = self.get_body_com('obj'+str(self.target_mug_id))
        elif self.now_task == TASKS.DRAWER_PLACE:
            if self.task_step == 0:
                self._target_pos = self.get_body_com('drawer_link'+str(self.target_drawer_id)) + np.array([-.01, -.01, -.09]) + np.array([.0, .0, .038])
                self.obj_init_pos = self.get_body_com('obj'+str(self.target_mug_id))
        elif self.now_task == TASKS.DESK_PICK:
            if self.task_step == 0:
                self._target_pos = self._get_mug_pick_pos(self.target_mug_id)
                self._target_pos[2] = 0.4
                self.quat = np.array([1., 0., 0., 0.])
                self.succeed = False
                self.obj_init_pos = self.get_body_com('obj'+str(self.target_mug_id))
        elif self.now_task == TASKS.DESK_PLACE:
            if self.task_step == 0:
                self._target_pos = self._random_init_point(self.target_mug_id)
                self.quat = np.array([1., 0., 0., 0.])
                self.succeed = False
                self.obj_init_pos = self.get_body_com('obj'+str(self.target_mug_id))
        elif self.now_task == TASKS.RESET_HAND:
            if self.task_step == 0:
                self._target_pos = np.array([0.0, 0.4, 0.4])
                self.succeed = False
                self.quat = np.array([1., 0., 0., 0.])
        else:
            raise NotImplementedError()

        reward, info = NAME2ENVS[self.now_task].evaluate_state(self, obs, action)

        info['task_name'] = self.now_task

        self.task_step += 1
        info['task_step'] = self.task_step
        self.after_success_cnt += info.get('after_success', False)
        if 'after_success' in info:
            info['after_success'] = self.after_success_cnt >= 10

        done = bool(info['success']) and bool(info.get('after_success', True))

        self._reset_state(info['success'])

        if done:
            done_task = self.task_list.pop(0)
            logger.info(f"{done_task} task done")
            self.task_step = 0
            self.after_success_cnt = 0
            logger.info(f'OLD STATES: {self.states}')
            self.target_states = change_state(parse_task(done_task),
                                              self.target_states)
            logger.info(f'NEW STATES: {self.states}')
            if self.random_generate_task:
                # TODO 下一步工作：如何随机生成任务
                self.random_generate_next_task()

        return reward, info

    def _random_init_point(self, pos=None):
        if pos is None:
            x_range = [-0.5, 0.5]
            y_range = [0.4, 0.9]
            forbid_list = [((drawer_pos[0]-0.25, drawer_pos[0]+0.25),
                        (drawer_pos[1]-0.4, drawer_pos[1]+0.2)) for drawer_pos in self.drawer_init_pos] + \
                        [((self.coffee_machine_init_pos[0]-0.25, self.coffee_machine_init_pos[0]+0.25),
                          (self.coffee_machine_init_pos[1]-0.3, self.coffee_machine_init_pos[1]+0.2))]
            for i in range(3):
                mug_pos = self.get_body_com('obj'+str(i))
                forbid_list.append(((mug_pos[0]-0.1, mug_pos[0]+0.1),
                                    (mug_pos[1]-0.1, mug_pos[1]+0.1)))
            x, y = random_grid_pos(x_range, y_range, forbid_list)
            pos = [x, y, 0]
        pos = np.array(pos)
        return pos
    
    def _get_mug_pick_pos(self, id):
        return self.get_body_com('obj'+str(id)) * np.array([1., 1., 0.]) + np.array([0., 0., 0.3])
    
    def set_random_generate_task(self, flag=True):
        self.random_generate_task = flag
        self.random_generate_next_task()

    def random_generate_next_task(self):
        total_tasks = deepcopy(list(NAME2ENVS.keys()))
        valid_tasks = list()
        valid_probs = list()
        for next_task in total_tasks:
            if check_task_cond(next_task, self.target_states):
                valid_tasks.append(next_task)
                valid_probs.append(TASK_RANDOM_PROBABILITY[next_task])
        self.task_list = random.choices(valid_tasks, weights=valid_probs, k=1)
        logger.info(f"random reset task list: {self.task_list}")

    @property
    def states(self) -> Dict[str, str]:
        return deepcopy(self._states)

    def read_states(self) -> str:
        return f'当前各个物体状态如下：{self.states}'.replace("'", '').replace('"', '')

    @property
    def target_states(self) -> dict[str, str]:
        return {
            STATE_KEYS.CUP: self.states[
                STATE_KEYS.CUP][self.target_mug_id],
            STATE_KEYS.DRAWER: self.states[
                STATE_KEYS.DRAWER][self.target_drawer_id],
            STATE_KEYS.COFFEE_MACHINE: self.states[
                STATE_KEYS.COFFEE_MACHINE][self.target_coffee_machine_id],
            STATE_KEYS.CUP_IN_DRAWER: self.states[
                STATE_KEYS.CUP_IN_DRAWER][self.target_mug_id],
            STATE_KEYS.DRAWER_CONTAINS_CUP: self.states[
                STATE_KEYS.DRAWER_CONTAINS_CUP][self.target_drawer_id],
            STATE_KEYS.HANDLE_OBJECT: self.states[STATE_KEYS.HANDLE_OBJECT],
        }

    @target_states.setter
    def target_states(self, states: dict[str, str]):
        self._states[STATE_KEYS.CUP][self.target_mug_id] = \
            states[STATE_KEYS.CUP]
        self._states[STATE_KEYS.DRAWER][self.target_drawer_id] = \
            states[STATE_KEYS.DRAWER]
        self._states[STATE_KEYS.COFFEE_MACHINE][self.target_coffee_machine_id] = \
            states[STATE_KEYS.COFFEE_MACHINE]
        self._states[STATE_KEYS.CUP_IN_DRAWER][self.target_mug_id] = \
            states[STATE_KEYS.CUP_IN_DRAWER]
        self._states[STATE_KEYS.DRAWER_CONTAINS_CUP][self.target_drawer_id] = \
            states[STATE_KEYS.DRAWER_CONTAINS_CUP]

    def _task_map(self):
        """
        支持的任务列表：
        '()coffee-button'                   按咖啡机按钮      
        '()coffee-pull'                     将咖啡机旁的杯子拿走【要求接完咖啡后必须拿走咖啡杯，若无要求则放桌上】
        '()coffee-push'                     将手中的杯子放到咖啡机旁
        '(color/pos_drawer)drawer-close'    将(color/pos_drawer)的抽屉关上
        '(color/pos_drawer)drawer-open'     将(color/pos_drawer)的抽屉打开
        '(color/pos_drawer)drawer-pick'     将(color/pos_drawer)的抽屉中的杯子取出
        '(color/pos_drawer)drawer-place'    将手中的杯子放到(color/pos_drawer)的抽屉中
        '(color/pos_mug)desk-pick'          将(color/pos_mug)的杯子从桌上拿起
        '()desk-place'                      将手中的杯子放到桌面上
        '()reset-hand'                      机械臂复位
        支持的描述：
        color: 物体颜色
        pos:   方位描述(left, right, mid)
        """
        now_task = self.task_list[0]
        target_item_pro, task_name = now_task.split(")")
        target_item_pro = target_item_pro[1:]
        if task_name in [TASKS.DRAWER_CLOSE, TASKS.DRAWER_OPEN, TASKS.DRAWER_PICK, TASKS.DRAWER_PLACE]:
            self._get_target_drawer(target_item_pro)
        if task_name == TASKS.DRAWER_PICK:
            self._get_target_mug_from_drawer()
        if task_name == TASKS.DESK_PICK:
            self._get_target_mug(target_item_pro)
        self.now_task = task_name
    
    def _get_target_drawer(self, item_pro):
        if item_pro == "right":
            self.target_drawer_id = 0
        elif item_pro == "mid":
            self.target_drawer_id = 1
        elif item_pro == "left":
            self.target_drawer_id = 2
        else:
            color = item_pro
            if color not in self.color2item_dict:
                raise ValueError(f"No item match color {color}")
            item = self.color2item_dict[color]
            if "drawer" not in item:
                raise ValueError(f"Color {color} miss match drawer but {item}")
            self.target_drawer_id = int(item[-1])
    
    def _get_target_mug(self, item_pro):
        mug_list = []
        for i in range(3):
            mug_list.append((self.get_body_com('obj'+str(i))[0], i))
        mug_list.sort(key=lambda x:x[0])
        if item_pro == "right":
            self.target_mug_id = mug_list[0][1]
        elif item_pro == "mid":
            self.target_mug_id = mug_list[1][1]
        elif item_pro == "left":
            self.target_mug_id = mug_list[2][1]
        else:
            color = item_pro
            if color not in self.color2item_dict:
                raise ValueError(f"No item match color {color}")
            item = self.color2item_dict[color]
            if "mug" not in item:
                raise ValueError(f"Color {color} miss match mug but {item}")
            self.target_mug_id = int(item[-1])
    
    def _get_target_mug_from_drawer(self):
        return self.target_states[STATE_KEYS.DRAWER_CONTAINS_CUP]

    @property
    def handle_object(self) -> dict[str, int]:
        return self.states[STATE_KEYS.HANDLE_OBJECT]

    @property
    def target_mug_id(self) -> int:
        return self.handle_object[STATE_KEYS.CUP]

    @target_mug_id.setter
    def target_mug_id(self, index: int):
        # TODO Check if the index is valid.
        self._states[STATE_KEYS.HANDLE_OBJECT][STATE_KEYS.CUP] = index

    @property
    def target_drawer_id(self) -> int:
        return self.handle_object[STATE_KEYS.DRAWER]

    @target_drawer_id.setter
    def target_drawer_id(self, index: int):
        # TODO Check if the index is valid.
        self._states[STATE_KEYS.HANDLE_OBJECT][STATE_KEYS.DRAWER] = index

    @property
    def target_coffee_machine_id(self) -> int:
        return self.handle_object[STATE_KEYS.COFFEE_MACHINE]

    @target_coffee_machine_id.setter
    def target_coffee_machine_id(self, index: int):
        # TODO Check if the index is valid.
        self._states[STATE_KEYS.HANDLE_OBJECT][
            STATE_KEYS.COFFEE_MACHINE] = index
