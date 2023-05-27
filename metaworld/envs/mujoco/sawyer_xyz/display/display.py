
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
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_shelf_place_display import SawyerShelfPlaceEnvV2Display
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
                                          check_if_state_valid,
                                          find_missing_task,
                                          TASKS,
                                          STATES,
                                          RGB_COLOR_LIST,
                                          QUAT_LIST,
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
    TASKS.BIN_PICK: SawyerBinPickEnvV2Display,
    TASKS.BIN_PLACE: SawyerBinPlaceEnvV2Display,
    TASKS.RESET_HAND: SawyerResetHandEnvV2Display,
}


logger = get_logger(__name__)


class SawyerEnvV2Display(
            SawyerCoffeeButtonEnvV2Display,
            SawyerCoffeePullEnvV2Display,
            SawyerCoffeePushEnvV2Display,
            SawyerDrawerCloseEnvV2Display,
            SawyerDrawerOpenEnvV2Display,
            SawyerDrawerPickEnvV2Display,
            SawyerDrawerPlaceEnvV2Display,
            SawyerDeskPickEnvV2Display,
            SawyerDeskPlaceEnvV2Display,
            SawyerBinPickEnvV2Display,
            SawyerBinPlaceEnvV2Display,
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
        self.task_step = 0
        self.task_done = True
        self._cup_machine_offset = 0.28

        self.random_generate_task = False
        self._states = {'cup': None, 'drawer': None}

        self.fix_reset_flag = False # 固定重置环境

    @property
    def model_name(self):
        return full_display_path_for('sawyer_xyz/_sawyer_display.xml')

    def _random_drawer_init_quat(self, index=None):
        init_quat_list = QUAT_LIST[:3]
        if index is None:
            index = random.randint(0, len(init_quat_list)-1)
        quat = np.array(init_quat_list[index])

        self.sim.model.body_quat[
            self.sim.model.body_name2id('drawer')
        ] = quat
        self.drawer_quat_index = index
        self.drawer_quat = quat

    def _random_drawer_init_pos(self, pos=None):
        # TODO
        if pos is None:
            if self.random_level == 1:
                pos_id = self.random_obj_list.index('drawer')
                if pos_id == 0:
                    x_range = [-0.45, -0.35]
                elif pos_id == 1:
                    x_range = [-0.05, 0.05]
                else:
                    x_range = [0.35, 0.45]
                y_range = [0.78, 0.85]
                x, y = random_grid_pos(x_range, y_range)
                pos = [x, y, 0]

        pos = np.array(pos)

        self.sim.model.body_pos[
            self.sim.model.body_name2id('drawer')
        ] = pos
        self.drawer_init_pos = pos
    
    def _random_drawer_init_open(self, open_f=None):
        maxDist = 0.16
        open_f = random.choice([True, False]) if open_f is None else open_f
        if open_f:
            qpos = self.data.qpos.flat.copy()
            qvel = self.data.qvel.flat.copy()
            addr_drawer = self.model.get_joint_qpos_addr('drawer_goal_slidey')
            qpos[addr_drawer] = - maxDist + (random.random() * 0.01)
            self.set_state(qpos, qvel)
            self._states['drawer'] = STATES.DRAWER_STATE_OPENED
        else:
            self._states['drawer'] = STATES.DRAWER_STATE_CLOSED
    
    def _random_coffee_machine_init_quat(self, index=None):
        init_quat_list = QUAT_LIST[:3]
        if index is None:
            index = random.randint(0, len(init_quat_list)-1)
        quat = np.array(init_quat_list[index])

        self.sim.model.body_quat[
            self.sim.model.body_name2id('coffee_machine')
        ] = quat
        self.coffee_machine_quat_index = index
        self.coffee_machine_quat = quat
    
    def _random_coffee_machine_init_pos(self, pos=None):
        # TODO
        if pos is None:
            if self.random_level == 1:
                pos_id = self.random_obj_list.index('coffee_machine')
                if pos_id == 0:
                    x_range = [-0.45, -0.35]
                elif pos_id == 1:
                    x_range = [-0.05, 0.05]
                else:
                    x_range = [0.35, 0.45]
                y_range = [0.75, 0.85]
                x, y = random_grid_pos(x_range, y_range)
                pos = [x, y, 0]

        pos = np.array(pos)

        self.sim.model.body_pos[
            self.sim.model.body_name2id('coffee_machine')
        ] = pos
        self.coffee_machine_init_pos = pos

    def _reset_button_offsets(self, pos=None):
        pos = 0 if pos is None else pos
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        addr = self.model.get_joint_qpos_addr('goal_slidey')
        qpos[addr] = pos
        self.set_state(qpos, qvel)

    @property
    def button_offsets(self):
        qpos = self.data.qpos.flat.copy()
        addr = self.model.get_joint_qpos_addr('goal_slidey')
        return qpos[addr]
    
    def _random_bin_init_pos(self, pos=None):
        if pos is None:
            if self.random_level == 1:
                pos_id = self.random_obj_list.index('bin')
                if pos_id == 0:
                    x_range = [-0.45, -0.35]
                elif pos_id == 1:
                    x_range = [-0.05, 0.05]
                else:
                    x_range = [0.35, 0.45]
                y_range = [0.55, 0.65]
                x, y = random_grid_pos(x_range, y_range)
                pos = [x, y, 0]

        pos = np.array(pos)

        self.sim.model.body_pos[
            self.sim.model.body_name2id('bin')
        ] = pos
        self.bin_init_pos = pos
    
    def _random_init_mug_pos(self, pos=None):
        # TODO
        if pos is None:
            if self.random_level == 1:
                x_range = [-0.5, 0.5]
                y_range = [0.4, 0.9]
                drawer_forbid = ((self.drawer_init_pos[0]-0.25, self.drawer_init_pos[0]+0.25),
                                 (self.drawer_init_pos[1]-0.4, self.drawer_init_pos[1]+0.2))
                machine_forbid = ((self.coffee_machine_init_pos[0]-0.25, self.coffee_machine_init_pos[0]+0.25),
                                  (self.coffee_machine_init_pos[1]-0.3, self.coffee_machine_init_pos[1]+0.2))
                bin_forbid = ((self.bin_init_pos[0]-0.25, self.bin_init_pos[0]+0.25),
                                (self.bin_init_pos[1]-0.2, self.bin_init_pos[1]+0.4))
                forbid_list = [drawer_forbid, machine_forbid, bin_forbid]
                x, y = random_grid_pos(x_range, y_range, forbid_list)
                pos = [x, y, 0]

        # pos = np.array(pos)
        pos_quat = np.array(pos + [1., 0., 0., 0.])
        self.mug_init_pos = pos
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        addr_mug = self.model.get_joint_qpos_addr('mug_obj')
        qpos[addr_mug[0]: addr_mug[1]] = pos_quat
        self.set_state(qpos, qvel)
    
    def _reset_mug_quat(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        addr_mug = self.model.get_joint_qpos_addr('mug_obj')
        # qpos[addr_mug[0]: addr_mug[1]] = pos_quat
        qpos[addr_mug[0]+3: addr_mug[0]+7] = np.array(np.array([1., 0., 0., 0.]))
        self.set_state(qpos, qvel)

    def _random_drawer_init(self, fix_flag):
        if self.random_level == 0:
            self._random_drawer_init_quat(0)
            self._random_drawer_init_pos([0.4, 0.85, 0.])
            self._random_drawer_init_open(True)
        elif self.random_level == 1:
            if fix_flag:
                self._random_drawer_init_quat(0)
                self._random_drawer_init_pos(self.drawer_init_pos)
                self._random_drawer_init_open(False)
            else:
                self._random_drawer_init_quat(0)
                self._random_drawer_init_pos()
                self._random_drawer_init_open(False)

        else:
            raise NotImplementedError()

    def _random_coffee_machine_init(self, fix_flag):
        if self.random_level == 0:
            self._random_coffee_machine_init_quat(0)
            self._random_coffee_machine_init_pos([0., 0.85, 0.])
        elif self.random_level == 1:
            if fix_flag:
                self._random_coffee_machine_init_quat(0)
                self._random_coffee_machine_init_pos(self.coffee_machine_init_pos)
            else:
                self._random_coffee_machine_init_quat(0)
                self._random_coffee_machine_init_pos()
        else:
            raise NotImplementedError()

    def _random_bin_init(self, fix_flag):
        if self.random_level == 0:
            self._random_bin_init_pos([-0.4, 0.85, 0.])
        elif self.random_level == 1:
            if fix_flag:
                self._random_bin_init_pos(self.bin_init_pos)
            else:
                self._random_bin_init_pos()
        else:
            raise NotImplementedError
    
    def _random_init_mug(self, fix_flag):
        if self.random_level == 0:
            self._random_init_mug_pos([-0.4, 0.65, 0.])
        else:
            if fix_flag:
                self._random_init_mug_pos(self.mug_init_pos)
            else:
                self._random_init_mug_pos()
                
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

        def set_model_rgba(model_name: str, rgb):
            if model_name == 'coffee_machine_body':
                model_name = ['coffee_machine_body1', 'coffee_machine_body2']
            elif model_name == 'mug':
                model_name = ['mug', 'handle']
            elif model_name == 'drawer':
                model_name = ['drawercase_tmp', 'drawer_tmp']
            elif model_name == 'shelf':
                model_name = ['shelf', 'shelf_supports']
            elif model_name == 'bin':
                # TODO
                model_name = ['binA']
            else:
                model_name = [model_name]

            rgba = list(rgb) + [1.]
            for name in model_name:
                self.sim.model.geom_rgba[
                    self.sim.model.geom_name2id(name)] = rgba

        # model_name_list = ['drawer', 'coffee_machine_body', 'shelf', 'mug']
        model_name_list = ['drawer', 'coffee_machine_body', 'bin', 'mug']
        rgb_list = random.sample(RGB_COLOR_LIST, len(model_name_list))
        for model_name, rgb in zip(model_name_list, rgb_list):
            set_model_rgba(model_name, rgb)

    def reset_model(self):
        """
        random_level:   0 - 固定物体位置
                        1 - 不改变drawer/coffee_machine/shelf朝向 随机三个物体的左中右 
        """

        self.random_level = 1

        self.task_list = []
        self.task_step = 0
        self.after_success_cnt = 0
        self.task_done = True

        
        self._random_table_and_floor()
            # self.random_obj_list = ['drawer', 'coffee_machine', 'shelf']
        if not self.fix_reset_flag:
            self.random_obj_list = ['drawer', 'coffee_machine', 'bin']
            random.shuffle(self.random_obj_list)

        self._random_drawer_init(self.fix_reset_flag)
        self._random_coffee_machine_init(self.fix_reset_flag)
        # self._random_shelf_init()
        self._random_bin_init(self.fix_reset_flag)
        self._random_init_mug(self.fix_reset_flag)

        self._random_init_color()
            
        self._random_init_hand_pos([0, 0.4, 0.4])

        self._target_pos = np.zeros(3)
        
        self.obj_init_pos = self._get_pos_objects()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        self._states['cup'] = STATES.CUP_STATE_DESK
        check_if_state_valid(self.states)
        logger.info(f"self.obj_init_pos: {self.obj_init_pos}")
        return self._get_obs()
    
    def fix_reset(self, flag=True):
        self.fix_reset_flag = flag

    def _get_pos_objects(self):
        if not hasattr(self, 'task_list') or len(self.task_list) == 0 or self.task_list[0] is None:
            return np.zeros(3)
        now_task = self.task_list[0]
        if now_task == TASKS.DRAWER_CLOSE:
            self.quat_index = self.drawer_quat_index
        elif now_task == TASKS.DRAWER_OPEN:
            self.quat_index = self.drawer_quat_index
        if now_task in NAME2ENVS.keys():
            results = NAME2ENVS[now_task]._get_pos_objects(self)
        else:
            raise NotImplementedError()
        return results

    def _get_quat_objects(self):
        if not hasattr(self, 'task_list') or len(self.task_list) == 0 or self.task_list[0] is None:
            return np.zeros(4)
        now_task = self.task_list[0]
        if now_task == TASKS.COFFEE_BUTTON:
            self.quat = self.coffee_machine_quat
        if now_task in NAME2ENVS.keys():
            results = NAME2ENVS[now_task]._get_quat_objects(self)
        else:
            raise NotImplementedError()
        return results
    
    def _check_task_list(self, task_list):
        for task in task_list:
            if task not in self.TASK_LIST:
                raise ValueError(f"Task name error: no task named {task}")
        
    def reset_task_list(self, task_list):
        self._check_task_list(task_list)
        self.task_list = task_list
        self.task_step = 0
        self.task_done = False
    
    def append_task_list(self, task_list):
        self._check_task_list(task_list)
        self.task_list += task_list
    
    def _judge_grab(self):
        # pos_mug = deepcopy(self.get_body_com('obj'))
        # if not hasattr(self, 'last_pos_mug'):
        #     flag = True
        # else:
        #     print(self.last_pos_mug)
        #     print(pos_mug)
        #     print(np.linalg.norm(self.last_pos_mug[:2] - pos_mug[:2]))
        #     flag = np.linalg.norm(self.last_pos_mug[:2] - pos_mug[:2]) <= 0.001
        # self.last_pos_mug = pos_mug
        # return flag

        # pos_hand = self.get_endeff_pos()

        # finger_right, finger_left = (
        #     self._get_site_pos('rightEndEffector'),
        #     self._get_site_pos('leftEndEffector')
        # )
        # gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        # gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)

        # print("pos_mug", self.get_body_com('obj'))
        # pos_mug = self.get_body_com('obj') + np.array([0., 0., 0.07])
        # print("gripper_distance_apart:", np.linalg.norm(pos_hand - pos_mug), gripper_distance_apart)
        # flag = (np.linalg.norm(pos_hand[:2] - pos_mug[:2]) <= 0.02) and (gripper_distance_apart < 0.57)
        # return flag

        pos_mug = self.get_body_com('obj')
        pos_bin = self.bin_init_pos
        pos_drawer = self.get_body_com('drawer_link')
        if np.linalg.norm(pos_bin[:2] - pos_mug[:2]) <= 0.1:
            return pos_mug[2] > 0.01
        elif np.linalg.norm(pos_drawer[:2] - pos_mug[:2]) <= 0.1:
            return pos_mug[2] > 0.08
        return pos_mug[2] > 0.005
    
    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        # print("pos_mug", self.get_body_com('obj'))
        # self._reset_mug_quat()
        if not self._judge_grab():
            self._reset_mug_quat()
        if len(self.task_list) == 0:
            self.task_step = 0
            info = {
                'success': float(False),
                'task_name': None,
                'task_step': 0,
            }
            return 0., info

        now_task = self.task_list[0]
        if now_task == None:
            info = {
                'success': float(False),
                'task_name': None,
                'task_step': self.task_step,
            }
            self.task_step += 1
            if self.task_step >= self.rest_step:
                self.task_list.pop(0)
                self.task_step = 0
            return 0., info

        if self.task_step == 0:
            if not check_task_cond(now_task, self.states):
                warnings.warn(f'Task {now_task} is invalid for state: '
                              f'{self.states}.')
                missing_task = find_missing_task(now_task, self.states)
                if missing_task is None:
                    warnings.warn(f'Cannot find the missing task, stopped.')
                    self.task_step = 0
                    info = {
                        'success': float(False),
                        'task_name': None,
                        'task_step': 0,
                    }
                    return 0., info
                else:
                    warnings.warn(f'Find a potential missing task '
                                  f'{missing_task}, execute it first.')
                    self.task_list.insert(0, missing_task)
                    now_task = missing_task
            self._reset_button_offsets()
            logger.info(f'TaskList: {self.task_list}')

        if now_task == TASKS.COFFEE_BUTTON:
            if self.task_step == 0:
                self.max_dist = 0.09
                if self.coffee_machine_quat_index == 0:
                    pos_button = self.coffee_machine_init_pos + np.array([.0, -self._cup_machine_offset, .3])
                    self._target_pos = pos_button + np.array([.0, self.max_dist, .0])
                elif self.coffee_machine_quat_index == 1:
                    pos_button = self.coffee_machine_init_pos + np.array([self._cup_machine_offset, .0, .3])
                    self._target_pos = pos_button + np.array([-self.max_dist, .0, .0])
                elif self.coffee_machine_quat_index == 2:
                    pos_button = self.coffee_machine_init_pos + np.array([-self._cup_machine_offset, .0, .3])
                    self._target_pos = pos_button + np.array([self.max_dist, .0, .0])
                else:
                    if hasattr(self, 'coffee_machine_quat_index'):
                        raise ValueError(f"Error coffee_machine_quat_index: {self.coffee_machine_quat_index}")
                    else:
                        raise ValueError(f"Error no coffee_machine_quat_index")
                self.quat = self.coffee_machine_quat
                self.succeed = False
        elif now_task == TASKS.COFFEE_PULL:
            if self.task_step == 0:
                self.max_dist = 0.03
                assert hasattr(self, 'mug_init_pos')
                self._target_pos = self._get_mug_pick_pos()
                self.quat = self.coffee_machine_quat
                self.succeed = False
        elif now_task == TASKS.COFFEE_PUSH:
            if self.task_step == 0:
                self.max_dist = 0.03
                if self.coffee_machine_quat_index == 0:
                    pos_goal = self.coffee_machine_init_pos + np.array([.0, -self._cup_machine_offset, .0])
                    self._target_pos = pos_goal
                elif self.coffee_machine_quat_index == 1:
                    pos_goal = self.coffee_machine_init_pos + np.array([self._cup_machine_offset, .0, .0])
                    self._target_pos = pos_goal
                elif self.coffee_machine_quat_index == 2:
                    pos_goal = self.coffee_machine_init_pos + np.array([-self._cup_machine_offset, .0, .0])
                    self._target_pos = pos_goal
                else:
                    if hasattr(self, 'coffee_machine_quat_index'):
                        raise ValueError(f"Error coffee_machine_quat_index: {self.coffee_machine_quat_index}")
                    else:
                        raise ValueError(f"Error no coffee_machine_quat_index")
                self.quat = self.coffee_machine_quat
                self.succeed = False
        elif now_task == TASKS.DRAWER_CLOSE:
            if self.task_step == 0:
                if self.drawer_quat_index == 0:
                    self._target_pos = self.get_body_com('drawer') + np.array([.0, -.16, .09])
                elif self.drawer_quat_index == 1:
                    self._target_pos = self.get_body_com('drawer') + np.array([+.16, .0, .09])
                elif self.drawer_quat_index == 2:
                    self._target_pos = self.get_body_com('drawer') + np.array([-.16, .0, .09])
                else:
                    self._target_pos = self.get_body_com('drawer') + np.array([.0, +.16, .09])
                self.obj_init_pos = self._get_pos_objects()
        elif now_task == TASKS.DRAWER_OPEN:
            if self.task_step == 0:
                self.maxDist = 0.16
                if self.drawer_quat_index == 0:
                    self._target_pos = self.get_body_com('drawer') + np.array([.0, -.16 - self.maxDist, .09])
                    self._handle_pos_init = self._target_pos + np.array([.0, self.maxDist, .0])
                elif self.drawer_quat_index == 1:
                    self._target_pos = self.get_body_com('drawer') + np.array([+.16 + self.maxDist, .0, .09])
                    self._handle_pos_init = self._target_pos + np.array([-self.maxDist, 0., .0])
                elif self.drawer_quat_index == 2:
                    self._target_pos = self.get_body_com('drawer') + np.array([-.16 - self.maxDist, .0, .09])
                    self._handle_pos_init = self._target_pos + np.array([self.maxDist, 0., .0])
                else:
                    self._target_pos = self.get_body_com('drawer') + np.array([.0, +.16 + self.maxDist, .09])
                    self._handle_pos_init = self._target_pos + np.array([.0, -self.maxDist, .0])
        elif now_task == TASKS.DRAWER_PICK:
            if self.task_step == 0:
                if self.drawer_quat_index == 0:
                    self._target_pos = self.get_body_com('obj') + np.array([.0, .0, .3])
                elif self.drawer_quat_index == 1:
                    self._target_pos = self.get_body_com('obj') + np.array([.0, .0, .3])
                elif self.drawer_quat_index == 2:
                    self._target_pos = self.get_body_com('obj') + np.array([.0, .0, .3])
                else:
                    self._target_pos = self.get_body_com('obj') + np.array([.0, .0, .3])
                self.obj_init_pos = self.get_body_com('obj')
        elif now_task == TASKS.DRAWER_PLACE:
            # print("obj:", self.get_body_com('obj'))
            # print("drawer:", self.get_body_com('drawer_link'))
            if self.task_step == 0:
                if self.drawer_quat_index == 0:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([-.01, -.01, -.09]) + np.array([.0, .0, .038])
                elif self.drawer_quat_index == 1:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([+.01, .0, -.09]) + np.array([.0, .0, .038])
                elif self.drawer_quat_index == 2:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([-.01, .0, -.09]) + np.array([.0, .0, .038])
                else:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([.0, +.01, -.09]) + np.array([.0, .0, .038])
                self.obj_init_pos = self.get_body_com('obj')
        elif now_task == TASKS.DESK_PICK:
            if self.task_step == 0:
                self._target_pos = self._get_mug_pick_pos()
                self._target_pos[2] = 0.4
                self.quat = self.coffee_machine_quat
                self.succeed = False
                self.obj_init_pos = self.get_body_com('obj')
        elif now_task == TASKS.DESK_PLACE:
            if self.task_step == 0:
                self._target_pos = self._random_init_point()
                self.quat = self.coffee_machine_quat
                self.succeed = False
                self.obj_init_pos = self.get_body_com('obj')
        elif now_task == TASKS.BIN_PICK:
            if self.task_step == 0:
                self._target_pos = self._get_mug_pick_pos()
                self._target_pos[2] = 0.4
                self.quat = self.coffee_machine_quat
                self.succeed = False
                self.obj_init_pos = self.get_body_com('obj')
        elif now_task == TASKS.BIN_PLACE:
            if self.task_step == 0:
                self._target_pos = self.get_body_com('bin')
                self.succeed = False
                self.quat = self.coffee_machine_quat
                self.obj_init_pos = self.get_body_com('obj')
        elif now_task == TASKS.RESET_HAND:
            if self.task_step == 0:
                self._target_pos = np.array([0.0, 0.4, 0.4])
                self.succeed = False
                self.quat = self.coffee_machine_quat
        else:
            raise NotImplementedError()

        reward, info = NAME2ENVS[now_task].evaluate_state(self, obs, action)

        info['task_name'] = now_task

        self.task_step += 1
        info['task_step'] = self.task_step
        self.after_success_cnt += info.get('after_success', False)
        if 'after_success' in info:
            info['after_success'] = self.after_success_cnt >= 10

        done = bool(info['success']) and bool(info.get('after_success', True))
        if done:
            done_task = self.task_list.pop(0)
            logger.info(f"{done_task} task done")
            self.task_step = 0
            self.after_success_cnt = 0
            logger.info(f'Finished Task: {done_task}')
            logger.info(f'Old States: {self.states}')
            self._states = change_state(done_task, self.states)
            logger.info(f'New States: {self.states}')
            if self.random_generate_task:
                self.random_generate_next_task()

        return reward, info

    def _random_init_point(self, pos=None):
        if pos is None:
            x_range = [-0.5, 0.5]
            y_range = [0.4, 0.9]
            drawer_forbid = (
                (self.drawer_init_pos[0]-0.25, self.drawer_init_pos[0]+0.25),
                (self.drawer_init_pos[1]-0.4, self.drawer_init_pos[1]+0.2))
            machine_forbid = (
                (self.coffee_machine_init_pos[0]-0.25, self.coffee_machine_init_pos[0]+0.25),
                (self.coffee_machine_init_pos[1]-0.3, self.coffee_machine_init_pos[1]+0.2))
            bin_forbid = (
                (self.bin_init_pos[0]-0.25, self.bin_init_pos[0]+0.25),
                (self.bin_init_pos[1]-0.2, self.bin_init_pos[1]+0.4))
            forbid_list = [drawer_forbid, machine_forbid, bin_forbid]
            x, y = random_grid_pos(x_range, y_range, forbid_list)
            pos = [x, y, 0]
        pos = np.array(pos)
        return pos
    
    def _get_mug_pick_pos(self):
        return self.get_body_com('obj') * np.array([1., 1., 0.]) + np.array([0., 0., 0.3])
    
    def set_random_generate_task(self, flag=True):
        self.random_generate_task = flag
        self.random_generate_next_task()

    def random_generate_next_task(self):
        total_tasks = deepcopy(list(NAME2ENVS.keys()))
        valid_tasks = list()
        valid_probs = list()
        for next_task in total_tasks:
            if check_task_cond(next_task, self.states):
                valid_tasks.append(next_task)
                valid_probs.append(TASK_RANDOM_PROBABILITY[next_task])
        self.task_list = random.choices(valid_tasks, weights=valid_probs, k=1)
        logger.info(f"random reset task list: {self.task_list}")

    @property
    def states(self) -> Dict[str, str]:
        return deepcopy(self._states)

    def read_states(self) -> str:
        return f'当前各个物体状态如下：{self.states}'.replace("'", '').replace('"', '')
