import numpy as np
from gym.spaces import Box
import random

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_display_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import _assert_task_is_set

from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_base import SawyerXYZEnvDisplay

from metaworld.envs.display_utils import RGB_COLOR_LIST, QUAT_LIST
from metaworld.envs.display_utils import random_grid_pos

class SawyerDrawerOpenEnvV2Display(SawyerXYZEnvDisplay):
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

        self.maxDist = 0.15
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_display_path_for('sawyer_xyz/sawyer_drawer.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            gripper_error,
            gripped,
            handle_error,
            caging_reward,
            opening_reward,
            success
        ) = self.compute_reward_drawer_open(action, obs)

        info = {
            'success': float(success),
            'near_object': float(gripper_error <= 0.03),
            'grasp_success': float(gripped > 0),
            'grasp_reward': caging_reward,
            'in_place_reward': opening_reward,
            'obj_to_target': handle_error,
            'unscaled_reward': reward,
        }

        info['after_success'] = self._get_after_success_drawer_open(info)
        after_reward = self._get_after_reward_drawer_open(info)

        return reward + after_reward, info
    
    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        self.set_state(qpos, qvel)

    # def _get_id_main_object(self):
    #     return self.unwrapped.model.geom_name2id('objGeom')

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
        self._set_obj_xyz(-0.1 * random.random())

        self._random_init_color()
        self._random_init_hand_pos()

        self._get_quat_index()
        if self.quat_index == 0:
            self._target_pos = self.get_body_com('drawer') + np.array([.0, -.16 - self.maxDist, .09])
            self._handle_pos_init = self._target_pos + np.array([.0, self.maxDist, .0])
        elif self.quat_index == 1:
            self._target_pos = self.get_body_com('drawer') + np.array([+.16 + self.maxDist, .0, .09])
            self._handle_pos_init = self._target_pos + np.array([-self.maxDist, 0., .0])
        elif self.quat_index == 2:
            self._target_pos = self.get_body_com('drawer') + np.array([-.16 - self.maxDist, .0, .09])
            self._handle_pos_init = self._target_pos + np.array([self.maxDist, 0., .0])
        else:
            self._target_pos = self.get_body_com('drawer') + np.array([.0, +.16 + self.maxDist, .09])
            self._handle_pos_init = self._target_pos + np.array([.0, -self.maxDist, .0])

        self.prev_obs = self._get_curr_obs_combined_no_goal()
        # self._reset_hand()
        # self.prev_obs = self._get_curr_obs_combined_no_goal()

        # Compute nightstand position
        # self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
        #     else self.init_config['obj_init_pos']
        # Set mujoco body to computed position
        # self.sim.model.body_pos[self.model.body_name2id(
        #     'drawer'
        # )] = self.obj_init_pos
        # Set _target_pos to current drawer position (closed) minus an offset
        # self._target_pos = self.obj_init_pos + np.array([.0, -.16 - self.maxDist, .09])

        return self._get_obs()

    def compute_reward_drawer_open(self, action, obs):
        gripper = obs[:3]
        handle = obs[4:7]
        # print("drawer_link_len:", self.sim.data.qpos[9])
        # print('drawer:', self.get_body_com('drawer'))
        # print('drawer_link:', self.get_body_com('drawer_link'))
        # print('_target_pos:', self._target_pos)
        # print('_handle_pos_init:', self._handle_pos_init)
        handle_error = np.linalg.norm(handle - self._target_pos)

        reward_for_opening = reward_utils.tolerance(
            handle_error,
            bounds=(0, 0.02),
            margin=self.maxDist,
            sigmoid='long_tail'
        )

        # handle_pos_init = self._target_pos + np.array([.0, self.maxDist, .0])
        handle_pos_init = self._handle_pos_init
        # Emphasize XY error so that gripper is able to drop down and cage
        # handle without running into it. By doing this, we are assuming
        # that the reward in the Z direction is small enough that the agent
        # will be willing to explore raising a finger above the handle, hook it,
        # and drop back down to re-gain Z reward
        scale = np.array([3., 3., 1.])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - self.init_tcp) * scale

        reward_for_caging = reward_utils.tolerance(
            np.linalg.norm(gripper_error),
            bounds=(0, 0.01),
            margin=np.linalg.norm(gripper_error_init),
            sigmoid='long_tail'
        )

        reward = reward_for_caging + reward_for_opening
        reward *= 5.0

        success = False
        drawer_link_pos = self.get_body_com('drawer_link')
        if self.quat_index == 0:
            success = drawer_link_pos[1] <= self._handle_pos_init[1]
        elif self.quat_index == 1:
            success = drawer_link_pos[0] >= self._handle_pos_init[0]
        elif self.quat_index == 2:
            success = drawer_link_pos[0] <= self._handle_pos_init[0]
        else:
            success = drawer_link_pos[1] >= self._handle_pos_init[1]
        
        if success:
            reward = 10.

        return (
            reward,
            np.linalg.norm(handle - gripper),
            obs[3],
            handle_error,
            reward_for_caging,
            reward_for_opening,
            success
        )
    
    def _get_after_success_drawer_open(self, info):
        hand_pos = self.get_endeff_pos()
        return info['success'] and (hand_pos[2] >= 0.29)

    def _get_after_reward_drawer_open(self, info):
        if not info['success']:
            return 0
        else:
            hand_pos = self.get_endeff_pos()
            return 2 * hand_pos[2] / 0.3
