import numpy as np
import random
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_display_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

from metaworld.envs.display_utils import RGB_COLOR_LIST, QUAT_LIST


class SawyerMugEnvV2Display(SawyerXYZEnv):
    _TARGET_RADIUS = 0.01
    def __init__(self, init_pos=None):

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

        if init_pos is None:
            self.init_pos = np.array([0., 0.9, 0.0], dtype=np.float32)
        else:
            self.init_pos = np.array(init_pos, dtype=np.float32)

    @property
    def model_name(self):
        return full_display_path_for('sawyer_xyz/sawyer_mug.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward,
        tcp_to_obj,
        _,
        target_to_obj,
        object_grasped,
        in_place) = self.compute_reward(action, obs)

        info = {
            'success': float(target_to_obj <= self.TARGET_RADIUS+0.005),
            # 'success': float(False),
            'near_object': float(tcp_to_obj <= 0.01),
            'grasp_success': 1.,
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }

        info['after_success'] = self._get_after_success(info)
        after_reward = self._get_after_reward(info)

        return reward + after_reward, info

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _get_quat_objects(self):
        return self.get_body_quat('mug')
    
    def _random_init_mug_pos(self):
        pos = self.init_pos
        return pos
    
    def _set_obj_xyz(self, mug_pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        mug_addr = self.model.get_joint_qpos_addr('mug_obj') # 后四位为quat
        qpos[mug_addr[0]:mug_addr[0]+3] = mug_pos
        self.set_state(qpos, qvel)

    def reset_model(self):
        # self._reset_hand()
        self.obj_init_pos = self._random_init_mug_pos()
        self._set_obj_xyz(self.obj_init_pos)
        self._reset_hand()
        
        self._target_pos = self.obj_init_pos
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        print("mug_pos:", self.get_body_com('obj'))

        return self._get_obs()

    def compute_reward(self, action, obs):

        return (0.,
               0.,
               0.,
               0.,
               0.,
               0.)
    
    def _get_after_success(self, info):
        return False

    def _get_after_reward(self, info):
        return 0.

