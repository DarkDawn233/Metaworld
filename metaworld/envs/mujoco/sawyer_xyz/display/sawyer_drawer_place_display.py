import numpy as np
import random
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_display_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import _assert_task_is_set

from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_base import SawyerXYZEnvDisplay

from metaworld.envs.display_utils import RGB_COLOR_LIST, QUAT_LIST
from metaworld.envs.display_utils import random_grid_pos


class SawyerDrawerPlaceEnvV2Display(SawyerXYZEnvDisplay):
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
        return full_display_path_for('sawyer_xyz/sawyer_drawer_mug.xml')

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
    
    def _set_obj_xyz(self, pos_drawer_link, pos_mug):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        addr_drawer = self.model.get_joint_qpos_addr('drawer_goal_slidey')
        qpos[addr_drawer] = pos_drawer_link
        addr_mug = self.model.get_joint_qpos_addr('mug_obj')
        qpos[addr_mug[0]: addr_mug[0] + 3] = pos_mug
        self.set_state(qpos, qvel)

    def _get_pos_objects(self):
        return np.hstack((
            self.get_body_com('obj'),
            self.get_body_com('drawer_link')
        ))

    def _get_quat_objects(self):
        return np.hstack((
            self.get_body_quat('mug'),
            self.get_body_quat('drawer_link')
        ))
    
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
        if self.drawer_quat_index == 0:
            x_range = [-0.45, 0.45]
            y_range = [0.65, 0.85]
        elif self.drawer_quat_index == 1:
            x_range = [-0.45, 0.25]
            y_range = [0.4, 0.8]
        elif self.drawer_quat_index == 2:
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
    
    def _random_init_mug_pos(self, pos=None):
        if pos is None:
            x_range = [-0.5, 0.5]
            y_range = [0.4, 0.9]
            xx, yy, _ = self.drawer_init_pos
            if self.drawer_quat_index == 0:
                forbid_zone = ((xx-0.2, xx+0.2), (yy-0.4, yy+0.2))
            elif self.drawer_quat_index == 1:
                forbid_zone = ((xx-0.2, xx+0.4), (yy-0.2, yy+0.2))
            elif self.drawer_quat_index == 1:
                forbid_zone = ((xx-0.4, xx+0.2), (yy-0.2, yy+0.2))
            else:
                forbid_zone = ((xx-0.2, xx+0.2), (yy-0.2, yy+0.4))
            forbid_list = [forbid_zone]
            x, y = random_grid_pos(x_range, y_range, forbid_list)
            pos = [x, y, 0.]
        return np.array(pos)

    def _random_drawer_init_quat(self, index=None):
        init_quat_list = QUAT_LIST[:4]
        if index is None:
            index = random.randint(0, len(init_quat_list)-1)
        self.drawer_quat_index = index
        quat = np.array(init_quat_list[index])
        self.sim.model.body_quat[
            self.sim.model.body_name2id('drawer')
        ] = quat
        return quat 

    def _random_init_color(self):
        rgbs = random.sample(RGB_COLOR_LIST, 2)
        rgbas = [np.array(list(rgb) + [1.]) for rgb in rgbs]

        self.sim.model.geom_rgba[
            self.sim.model.geom_name2id('drawercase_tmp')
        ] = rgbas[0]
        self.sim.model.geom_rgba[
            self.sim.model.geom_name2id('drawer_tmp')
        ] = rgbas[0]

        self.sim.model.geom_rgba[
            self.sim.model.geom_name2id('mug')
        ] = rgbas[1]
        self.sim.model.geom_rgba[
            self.sim.model.geom_name2id('handle')
        ] = rgbas[1]
    
    def _get_drawer_quat_index(self):
        eps = 1e-5
        quat = self.get_body_quat('drawer_link')
        quat_list = QUAT_LIST[:4]
        for i in range(len(quat_list)):
            if np.sum((quat - np.array(quat_list[i])) ** 2) < eps:
                self.drawer_quat_index = i
                break

    def reset_model(self):

        self._random_table_and_floor()

        self.drawer_init_quat = self._random_drawer_init_quat()
        self._get_drawer_quat_index()
        self.drawer_init_pos = self._random_init_drawer_pos()
        self.mug_init_pos = self._random_init_mug_pos()
        # print("mug_pos:", self.mug_init_pos)
        # self._set_obj_xyz(-self.maxDist * random.random())
        self._set_obj_xyz(-self.maxDist + (random.random() * 0.01), self.mug_init_pos)
        # self._target_pos = self.get_body_com('drawer_link')

        self._random_init_color()
        self._random_init_hand_pos()

        
        if self.drawer_quat_index == 0:
            self._target_pos = self.get_body_com('drawer_link') + np.array([.0, -.01, -.09])
        elif self.drawer_quat_index == 1:
            self._target_pos = self.get_body_com('drawer_link') + np.array([+.01, .0, -.09])
        elif self.drawer_quat_index == 2:
            self._target_pos = self.get_body_com('drawer_link') + np.array([-.01, .0, -.09])
        else:
            self._target_pos = self.get_body_com('drawer_link') + np.array([.0, +.01, -.09])
        
        self.obj_init_pos = self.get_body_com('obj')
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        return self._get_obs()

    def _gripper_caging_reward(self, action, obj_position):
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015
        tcp = self.tcp_center
        left_pad = self.get_body_com('leftpad')
        right_pad = self.get_body_com('rightpad')
        delta_object_y_left_pad = left_pad[1] - obj_position[1]
        delta_object_y_right_pad = obj_position[1] - right_pad[1]
        right_caging_margin = abs(abs(obj_position[1] - self.init_right_pad[1])
            - pad_success_margin)
        left_caging_margin = abs(abs(obj_position[1] - self.init_left_pad[1])
            - pad_success_margin)

        right_caging = reward_utils.tolerance(delta_object_y_right_pad,
                                bounds=(obj_radius, pad_success_margin),
                                margin=right_caging_margin,
                                sigmoid='long_tail',)
        left_caging = reward_utils.tolerance(delta_object_y_left_pad,
                                bounds=(obj_radius, pad_success_margin),
                                margin=left_caging_margin,
                                sigmoid='long_tail',)

        y_caging = reward_utils.hamacher_product(left_caging,
                                                 right_caging)

        # compute the tcp_obj distance in the x_z plane
        tcp_xz = tcp + np.array([0., -tcp[1], 0.])
        obj_position_x_z = np.copy(obj_position) + np.array([0., -obj_position[1], 0.])
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)

        # used for computing the tcp to object object margin in the x_z plane
        init_obj_x_z = self.obj_init_pos + np.array([0., -self.obj_init_pos[1], 0.])
        init_tcp_x_z = self.init_tcp + np.array([0., -self.init_tcp[1], 0.])
        tcp_obj_x_z_margin = np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin

        x_z_caging = reward_utils.tolerance(tcp_obj_norm_x_z,
                                bounds=(0, x_z_success_margin),
                                margin=tcp_obj_x_z_margin,
                                sigmoid='long_tail',)

        gripper_closed = min(max(0, action[-1]), 1)
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)

        gripping = gripper_closed if caging > 0.97 else 0.
        caging_and_gripping = reward_utils.hamacher_product(caging,
                                                            gripping)
        caging_and_gripping = (caging_and_gripping + caging) / 2
        return caging_and_gripping

    def compute_reward(self, action, obs):
        # print('drawer:', self.get_body_com('drawer'))
        # print('drawer_link:', self.get_body_com('drawer_link'))
        # print('obj_init_pos:', self.obj_init_pos)
        # print('target_pos:', self._target_pos)
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self._target_pos

        obj_to_target = np.linalg.norm(obj - target)
        tcp_to_obj = np.linalg.norm(obj - tcp)
        in_place_margin = (np.linalg.norm(self.obj_init_pos - target))

        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, self.TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        object_grasped = self._gripper_caging_reward(action, obj)
        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                    in_place)
        reward = in_place_and_object_grasped

        # if tcp_to_obj < 0.02 and (tcp_opened > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]):
        #     reward += 1. + 5. * in_place
        
        reward += 1. + 5. * in_place
        if obj_to_target < self.TARGET_RADIUS:
            reward = 10.

        return (reward,
               tcp_to_obj,
               tcp_opened,
               obj_to_target,
               object_grasped,
               in_place)
    
    def _get_after_success(self, info):
        hand_pos = self.get_endeff_pos()
        return info['success'] and (hand_pos[2] >= 0.29)

    def _get_after_reward(self, info):
        if not info['success']:
            return 0
        else:
            hand_pos = self.get_endeff_pos()
            return 2 * hand_pos[2] / 0.3

