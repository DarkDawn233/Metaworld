import numpy as np
import random
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_display_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import _assert_task_is_set

from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_base import SawyerXYZEnvDisplay

from metaworld.envs.display_utils import RGB_COLOR_LIST, QUAT_LIST
from metaworld.envs.display_utils import random_grid_pos




class SawyerDrawerCloseEnvV2Display(SawyerXYZEnvDisplay):
    _TARGET_RADIUS = 0.01
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.0)
        obj_high = (0.1, 0.9, 0.0)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0., 0.9, 0.0], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.3], dtype=np.float32),
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

        self.maxDist = 0.16
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_display_path_for('sawyer_xyz/sawyer_drawer.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward,
        tcp_to_obj,
        _,
        target_to_obj,
        object_grasped,
        in_place) = self.compute_reward_drawer_close(action, obs)

        info = {
            'success': float(target_to_obj <= self.TARGET_RADIUS+0.005),
            'near_object': float(tcp_to_obj <= 0.01),
            'grasp_success': 1.,
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }

        info['after_success'] = self._get_after_success_drawer_close(info)
        after_reward = self._get_after_reward_drawer_close(info)

        return reward + after_reward, info

    def _get_pos_objects(self):
        if not hasattr(self, 'quat_index'):
            return np.zeros(3)
        if self.quat_index == 0:
            pos = self.get_body_com('drawer_link') + np.array([.0, -.16, .05])
        elif self.quat_index == 1:
            pos = self.get_body_com('drawer_link') + np.array([+.16, .0, .05])
        elif self.quat_index == 2:
            pos = self.get_body_com('drawer_link') + np.array([-.16, .0, .05])
        else:
            pos = self.get_body_com('drawer_link') + np.array([.0, +.16, .05])
        return pos

    def _get_quat_objects(self):
        return self.get_body_quat('drawer_link')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        self.set_state(qpos, qvel)
    
    def _random_init_hand_pos(self, pos=None):
        if pos is None:
            x_range = [-0.5, 0.5]
            y_range = [0.25, 0.85]
            z_range = [0.29, 0.3]
            x = random.random() * (x_range[1] - x_range[0]) + x_range[0]
            y = random.random() * (y_range[1] - y_range[0]) + y_range[0]
            z = random.random() * (z_range[1] - z_range[0]) + z_range[0]
            pos = [x, y, z]
        self.hand_init_pos = pos
        self._reset_hand()

    def _random_init_drawer_pos(self, pos=None):
        if self.quat_index == 0:
            x_range = [-0.45, 0.45]
            y_range = [0.65, 0.85]
        elif self.quat_index == 1:
            x_range = [-0.45, 0.25]
            y_range = [0.4, 0.8]
        elif self.quat_index == 2:
            x_range = [-0.25, 0.45]
            y_range = [0.4, 0.8]
        else:
            x_range = [-0.45, 0.45]
            y_range = [0.35, 0.55]
        if pos is None:
            z = 0.
            x, y = random_grid_pos(x_range, y_range)
            # x = random.random() * (x_range[1] - x_range[0]) + x_range[0]
            # y = random.random() * (y_range[1] - y_range[0]) + y_range[0]
            pos = np.array([x, y, z])
        else:
            pos = np.array(pos)
        self.sim.model.body_pos[
            self.sim.model.body_name2id('drawer')
        ] = pos
        return pos

    def _random_init_quat(self, index=None):
        init_quat_list = QUAT_LIST[:4]
        if index is None:
            index = random.randint(0, len(init_quat_list)-1)
        self.quat_index = index
        quat = np.array(init_quat_list[index])
        self.sim.model.body_quat[
            self.sim.model.body_name2id('drawer')
        ] = quat
        return quat 

    def _random_init_color(self):
        rgb = random.choice(RGB_COLOR_LIST)
        rgba = np.array(list(rgb) + [1.])

        self.sim.model.geom_rgba[
            self.sim.model.geom_name2id('drawercase_tmp')
        ] = rgba
        self.sim.model.geom_rgba[
            self.sim.model.geom_name2id('drawer_tmp')
        ] = rgba
    
    def _get_quat_index(self):
        eps = 1e-5
        quat = self.get_body_quat('drawer_link')
        quat_list = QUAT_LIST[:4]
        for i in range(len(quat_list)):
            if np.sum((quat - np.array(quat_list[i])) ** 2) < eps:
                # print("index:", i)
                self.quat_index = i
                break

    def reset_model(self):

        self._random_table_and_floor()

        self.obj_init_quat = self._random_init_quat()
        self.obj_init_pos = self._random_init_drawer_pos()
        # self._set_obj_xyz(-self.maxDist * random.random())
        self._set_obj_xyz(-self.maxDist + (random.random() * 0.1))

        self._random_init_color()
        self._random_init_hand_pos()

        self._get_quat_index()
        if self.quat_index == 0:
            self._target_pos = self.get_body_com('drawer') + np.array([.0, -.16, .09])
        elif self.quat_index == 1:
            self._target_pos = self.get_body_com('drawer') + np.array([+.16, .0, .09])
        elif self.quat_index == 2:
            self._target_pos = self.get_body_com('drawer') + np.array([-.16, .0, .09])
        else:
            self._target_pos = self.get_body_com('drawer') + np.array([.0, +.16, .09])
        
        self.obj_init_pos = self._get_pos_objects()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        return self._get_obs()

    def compute_reward_drawer_close(self, action, obs):
        # print('drawer:', self.get_body_com('drawer'))
        # print('drawer_link:', self.get_body_com('drawer_link'))
        # print('obj_init_pos:', self.obj_init_pos)
        # print('target_pos:', self._target_pos)
        obj = obs[4:7]

        tcp = self.tcp_center
        target = self._target_pos.copy()

        target_to_obj = (obj - target)
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.obj_init_pos - target)
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid='long_tail',
        )

        handle_reach_radius = 0.005
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_reach_radius),
            margin=abs(tcp_to_obj_init-handle_reach_radius),
            sigmoid='gaussian',
        )
        gripper_closed = min(max(0, action[-1]), 1)

        reach = reward_utils.hamacher_product(reach, gripper_closed)
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        reward = in_place
        if target_to_obj <= self.TARGET_RADIUS+0.005:
            reward = 1.

        reward *= 10

        return (reward,
               tcp_to_obj,
               tcp_opened,
               target_to_obj,
               object_grasped,
               in_place)
    
    def _get_after_success_drawer_close(self, info):
        hand_pos = self.get_endeff_pos()
        return info['success'] and (hand_pos[2] >= 0.29)

    def _get_after_reward_drawer_close(self, info):
        if not info['success']:
            return 0
        else:
            hand_pos = self.get_endeff_pos()
            return 2 * hand_pos[2] / 0.3

