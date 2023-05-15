
from typing import Dict
import numpy as np
from gym.spaces import Box
import random

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_display_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import _assert_task_is_set

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
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_reset_display import SawyerResetEnvV2Display

from metaworld.envs.display_utils import RGB_COLOR_LIST, QUAT_LIST
from metaworld.envs.display_utils import random_grid_pos


NAME2ENVS: Dict[str, SawyerXYZEnvDisplay] = {
    'coffee-button': SawyerCoffeeButtonEnvV2Display,
    'coffee-pull': SawyerCoffeePullEnvV2Display,
    'coffee-push': SawyerCoffeePushEnvV2Display,
    'drawer-close': SawyerDrawerCloseEnvV2Display,
    'drawer-open': SawyerDrawerOpenEnvV2Display,
    'drawer-pick': SawyerDrawerPickEnvV2Display,
    'drawer-place': SawyerDrawerPlaceEnvV2Display,
    'desk-pick': SawyerDeskPickEnvV2Display,
    'desk-place': SawyerDeskPlaceEnvV2Display,
    'reset': SawyerResetEnvV2Display,
}


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
            SawyerResetEnvV2Display,
        ):

    TASK_LIST = [
        'coffee-button',
        'coffee-pull',
        'coffee-push',
        'drawer-close',
        'drawer-open',
        'drawer-pick',
        'drawer-place',
        'desk-pick',
        'desk-place',
        'reset',
    ]
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
        self.drawer_open_flag = False
        self.mug_in_drawer_flag = False
        # self.mug_grasped_flag = False

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
            self.drawer_open_flag = True
        else:
            self.drawer_open_flag = False
    
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
    
    def _random_shelf_init_quat(self, index=None):
        init_quat_list = QUAT_LIST[:3]
        if index is None:
            index = random.randint(0, len(init_quat_list)-1)
        quat = np.array(init_quat_list[index])

        self.sim.model.body_quat[
            self.sim.model.body_name2id('shelf')
        ] = quat
        self.shelf_quat_index = index
        self.shelf_quat = quat
    
    def _random_shelf_init_pos(self, pos=None):
        # TODO
        if pos is None:
            if self.random_level == 1:
                pos_id = self.random_obj_list.index('shelf')
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
            self.sim.model.body_name2id('shelf')
        ] = pos
        self.shelf_init_pos = pos
    
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
                shelf_forbid = ((self.shelf_init_pos[0]-0.25, self.shelf_init_pos[0]+0.25),
                                (self.shelf_init_pos[1]-0.2, self.shelf_init_pos[1]+0.2))
                forbid_list = [drawer_forbid, machine_forbid, shelf_forbid]
                x, y = random_grid_pos(x_range, y_range, forbid_list)
                pos = [x, y, 0]

        pos = np.array(pos)
        self.mug_init_pos = pos
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        addr_mug = self.model.get_joint_qpos_addr('mug_obj')
        qpos[addr_mug[0]: addr_mug[0] + 3] = pos
        self.set_state(qpos, qvel)

    def _random_drawer_init(self):
        if self.random_level == 0:
            self._random_drawer_init_quat(0)
            self._random_drawer_init_pos([0.4, 0.85, 0.])
            self._random_drawer_init_open(True)
        elif self.random_level == 1:
            self._random_drawer_init_quat(0)
            self._random_drawer_init_pos()
            self._random_drawer_init_open()
        else:
            raise NotImplementedError()

    def _random_coffee_machine_init(self):
        if self.random_level == 0:
            self._random_coffee_machine_init_quat(0)
            self._random_coffee_machine_init_pos([0., 0.85, 0.])
        elif self.random_level == 1:
            self._random_coffee_machine_init_quat(0)
            self._random_coffee_machine_init_pos()
        else:
            raise NotImplementedError()
    
    def _random_shelf_init(self):
        if self.random_level == 0:
            self._random_shelf_init_quat(0)
            self._random_shelf_init_pos([-0.4, 0.85, 0.])
        elif self.random_level == 1:
            self._random_shelf_init_quat(0)
            self._random_shelf_init_pos()
        else:
            raise NotImplementedError
    
    def _random_init_mug(self):
        if self.random_level == 0:
            self._random_init_mug_pos([-0.4, 0.65, 0.])
        else:
            self._random_init_mug_pos()
    
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
            else:
                model_name = [model_name]

            rgba = list(rgb) + [1.]
            for name in model_name:
                self.sim.model.geom_rgba[
                    self.sim.model.geom_name2id(name)] = rgba

        model_name_list = ['drawer', 'coffee_machine_body', 'shelf', 'mug']
        rgb_list = random.sample(RGB_COLOR_LIST, len(model_name_list))
        for model_name, rgb in zip(model_name_list, rgb_list):
            set_model_rgba(model_name, rgb)

    def reset_model(self):
        """
        random_level:   0 - 固定物体位置
                        1 - 不改变drawer/coffee_machine/shelf朝向 随机三个物体的左中右 
        """
        self.random_level = 1
        self._random_table_and_floor()
  
        self.random_obj_list = ['drawer', 'coffee_machine', 'shelf']
        random.shuffle(self.random_obj_list)

        self._random_drawer_init()
        self._random_coffee_machine_init()
        self._random_shelf_init()
        self._random_init_mug()

        self._random_init_color()
        self._random_init_hand_pos([0, 0.4, 0.2])

        self._target_pos = np.zeros(3)
        
        self.obj_init_pos = self._get_pos_objects()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        self.task_list = []
        self.task_step = 0
        self.task_done = True

        return self._get_obs()

    def _get_pos_objects(self):
        if not hasattr(self, 'task_list') or len(self.task_list) == 0:
            return np.zeros(3)
        now_task = self.task_list[0]
        if now_task == 'drawer-close':
            self.quat_index = self.drawer_quat_index
        elif now_task == 'drawer-open':
            self.quat_index = self.drawer_quat_index
        if now_task in NAME2ENVS.keys():
            results = NAME2ENVS[now_task]._get_pos_objects(self)
        else:
            raise NotImplementedError()
        return results

    def _get_quat_objects(self):
        if not hasattr(self, 'task_list') or len(self.task_list) == 0:
            return np.zeros(4)
        now_task = self.task_list[0]
        if now_task == 'coffee-button':
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
    
    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        if len(self.task_list) == 0:
            info = {
                'success': float(False),
                'task_name': None,
            }
            return 0., info

        now_task = self.task_list[0]
        if now_task == 'coffee-button':
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
            # reward, info = super(SawyerEnvV2Display, self).evaluate_state(obs, action)
        elif now_task == 'coffee-pull':
            if self.task_step == 0:
                self.max_dist = 0.03
                assert hasattr(self, 'mug_init_pos')
                # self._target_pos = self._random_init_point()
                self._target_pos = self._get_mug_pick_pos()
                self.quat = self.coffee_machine_quat
                self.succeed = False
            # reward, info = super(SawyerCoffeeButtonEnvV2Display, self).evaluate_state(obs, action)
        elif now_task == 'coffee-push':
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
            reward, info = super(SawyerCoffeePullEnvV2Display, self).evaluate_state(obs, action)
        elif now_task == 'drawer-close':
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
            # reward, info = super(SawyerCoffeePushEnvV2Display, self).evaluate_state(obs, action)
        elif now_task == 'drawer-open':
            if self.task_step == 0:
                self.maxDist = 0.15
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
            # reward, info = super(SawyerDrawerCloseEnvV2Display, self).evaluate_state(obs, action)
        elif now_task == 'drawer-pick':
            if self.task_step == 0:
                if self.drawer_quat_index == 0:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([.0, -.01, -.09]) + np.array([.0, .0, .3])
                elif self.drawer_quat_index == 1:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([+.01, .0, -.09]) + np.array([.0, .0, .3])
                elif self.drawer_quat_index == 2:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([-.01, .0, -.09]) + np.array([.0, .0, .3])
                else:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([.0, +.01, -.09]) + np.array([.0, .0, .3])
                self.obj_init_pos = self.get_body_com('obj')
            # reward, info = super(SawyerDrawerOpenEnvV2Display, self).evaluate_state(obs, action)
        elif now_task == 'drawer-place':
            # print(f"pos obj: {self.get_body_com('obj')}")
            # print(f"drawer link: {self.get_body_com('drawer_link')}")
            if self.task_step == 0:
                if self.drawer_quat_index == 0:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([.0, -.01, -.09])
                elif self.drawer_quat_index == 1:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([+.01, .0, -.09])
                elif self.drawer_quat_index == 2:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([-.01, .0, -.09])
                else:
                    self._target_pos = self.get_body_com('drawer_link') + np.array([.0, +.01, -.09])
                self.obj_init_pos = self.get_body_com('obj')
            # reward, info = super(SawyerDrawerPickEnvV2Display, self).evaluate_state(obs, action)
        elif now_task == 'desk-pick':
            if self.task_step == 0:
                # self._target_pos = self._random_init_point()
                self._target_pos = self._get_mug_pick_pos()
                self._target_pos[2] = 0.4
                self.quat = self.coffee_machine_quat
                self.succeed = False
            # reward, info = SawyerDeskPickEnvV2Display.evaluate_state(self, obs, action)
        elif now_task == 'desk-place':
            if self.task_step == 0:
                self._target_pos = self._random_init_point()
                self.quat = self.coffee_machine_quat
                self.succeed = False
            # reward, info = SawyerDeskPlaceEnvV2Display.evaluate_state(self, obs, action)
        elif now_task == 'reset':
            if self.task_step == 0:
                self._target_pos = np.array([0.0, 0.6, 0.4])
                self.succeed = False
                self.quat = self.coffee_machine_quat
            # reward, info = SawyerResetEnvV2Display.evaluate_state(self, obs, action)
        else:
            raise NotImplementedError()
        reward, info = NAME2ENVS[now_task].evaluate_state(self, obs, action)
        
        info['task_name'] = now_task
        
        self.task_step += 1
        info['task_step'] = self.task_step
        done = bool(info['success']) and bool(info.get('after_success', True))
        if done:
            done_task = self.task_list.pop(0)
            print(f"{done_task} task done")
            self.task_step = 0
            if self.random_generate_task:
                self.random_next_task(done_task)
        
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
            shelf_forbid = (
                (self.shelf_init_pos[0]-0.25, self.shelf_init_pos[0]+0.25),
                (self.shelf_init_pos[1]-0.2, self.shelf_init_pos[1]+0.2))
            forbid_list = [drawer_forbid, machine_forbid, shelf_forbid]
            x, y = random_grid_pos(x_range, y_range, forbid_list)
            pos = [x, y, 0]
        pos = np.array(pos)
        return pos
    
    def _get_mug_pick_pos(self):
        return self.get_body_com('obj') * np.array([1., 1., 0.]) + np.array([0., 0., 0.3])
    
    def set_random_generate_task(self, flag=True):
        self.random_generate_task = flag
        self.random_next_task(None)

    def random_next_task(self, last_task):
        if last_task == 'drawer-close':
            self.drawer_open_flag = False
        if last_task == 'drawer-open':
            self.drawer_open_flag = True
        if last_task == 'drawer-place':
            self.mug_in_drawer_flag = True
        if last_task == 'drawer-pick':
            self.mug_in_drawer_flag = False

        if last_task == 'coffee-push':
            optional_task = ['coffee-button']
        elif last_task == 'coffee-button':
            optional_task = ['coffee-pull']
        elif last_task in ['coffee-pull', 'drawer-pick', 'desk-pick']: # mug in hand
            optional_task = ['coffee-push', 'desk-place']
            if self.drawer_open_flag: # drawer is open
                optional_task.append('drawer-place')
        else:   # mug not in hand
            optional_task = ['reset']
            if self.drawer_open_flag: # drawer is open
                optional_task.append('drawer-close')
            if not self.drawer_open_flag: # drawer is closed
                optional_task.append('drawer-open')
            if self.drawer_open_flag and self.mug_in_drawer_flag: # drawer is open and mug in drawer
                optional_task.append('drawer-pick')
            if not self.mug_in_drawer_flag: # mug on desk
                optional_task.append('desk-pick')
            
            if last_task in optional_task:
                optional_task.remove(last_task)
        
        self.task_list = [random.choice(optional_task)]
        print(f"random reset task list: {self.task_list}")
            


            
