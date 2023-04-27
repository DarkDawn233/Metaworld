import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_display_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.display_utils import RGB_COLOR_LIST, QUAT_LIST
import random


class SawyerCoffeeButtonEnvV2Display(SawyerXYZEnv):

    def __init__(self):

        self.max_dist = 0.03

        hand_low = (-0.5, .4, 0.05)
        hand_high = (0.5, 1., 0.5)
        obj_low = (-0.1, 0.8, -.001)
        obj_high = (0.1, 0.9, +.001)
        # goal_low[3] would be .1, but objects aren't fully initialized until a
        # few steps after reset(). In that time, it could be .01
        goal_low = obj_low + np.array([-.001, -.22 + self.max_dist, .299])
        goal_high = obj_high + np.array([+.001, -.22 + self.max_dist, .301])

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.9, 0.28]),
            'obj_init_angle': 0.0,
            'hand_init_pos': np.array([0., .4, .2]),
        }
        self.goal = np.array([0, 0.78, 0.33])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.num_resets = 0
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self.succeed = False

    @property
    def model_name(self):
        return full_display_path_for('sawyer_xyz/sawyer_coffee.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed,
            success
        ) = self.compute_reward(action, obs)

        info = {
            'success': success,
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(tcp_open > 0),
            'grasp_reward': near_button,
            'in_place_reward': button_pressed,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        # info['after_success'] = self._get_after_success(info)
        if info['success']:
            self.succeed = True
            self._target_pos[2] = 0.37
            hand_pos = self.get_endeff_pos()
            info['after_success'] = info['success'] and (hand_pos[2] >= 0.39)
        if self.succeed:
            info['success'] = True
            reward = 10
            hand_pos = self.get_endeff_pos()
            info['after_success'] = info['success'] and (hand_pos[2] >= 0.39)
        # after_reward = self._get_after_reward(info)

        # return reward + after_reward, info
        return reward, info

    def _get_after_success(self, info):
        hand_pos = self.get_endeff_pos()
        return info['success'] and (hand_pos[2] >= 0.29)

    def _get_after_reward(self, info):
        if not info['success']:
            return 0
        else:
            self.succeed = True
            hand_pos = self.get_endeff_pos()
            return 2 * hand_pos[2] / 0.3

    @property
    def _target_site_config(self):
        return [('coffee_goal', self._target_pos)]

    def _get_id_main_object(self):
        return None

    def _get_pos_objects(self):
        return self._get_site_pos('buttonStart')

    def _get_quat_objects(self):
        if hasattr(self, 'quat'):
            return self.quat
        else:
            return np.array([1., 0., 0., 0.])

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        qpos[0:3] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    # def reset_model(self):
    #     self._reset_hand()
    #     self.sim.model.body_pos[self.model.body_name2id('coffee_machine')] = \
    #         self.init_config['obj_init_pos'] - np.array([0, 0, 0.0])
    #     self._target_pos = \
    #         (self.sim.model.site_pos[
    #             self.model.site_name2id('coffee_goal')]
    #          + self.sim.model.body_pos[
    #             self.model.body_name2id('coffee_machine')]
    #          + np.array([0, 0, 0]))
    #     # self.obj_init_pos = self.adjust_initObjPos(
    #     #     self.init_config['obj_init_pos'])
    #     self.obj_init_angle = self.init_config['obj_init_angle']
    #     print('First: ', self.obj_init_pos)
    #     if self.random_init:
    #         goal_pos = self._get_state_rand_vec()
    #         # while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
    #         #     goal_pos = self._get_state_rand_vec()
    #         base_coffee_machine_pos = goal_pos - np.array([0.2, 0.2, 0.0])
    #         self.obj_init_pos = np.concatenate((base_coffee_machine_pos[:2],
    #                                             [self.obj_init_pos[-1]]))
    #         print('Second: ', self.obj_init_pos)
    #         self.sim.model.body_pos[
    #             self.model.body_name2id('coffee_machine')] = \
    #                 base_coffee_machine_pos[-3:]
    #         self._target_pos = \
    #             (self.sim.model.site_pos[
    #                 self.model.site_name2id('coffee_goal')]
    #              + self.sim.model.body_pos[
    #                 self.model.body_name2id('coffee_machine')]
    #              + np.array([0, 0, 0]))

    #     self._set_obj_xyz(self.obj_init_pos)
    #     self.num_resets += 1

    #     return self._get_obs()

    # def reset_model(self):
    #     self._reset_hand()

    #     print('Running reset.')
    #     print(f'First init pos: {self.obj_init_pos}.')
    #     self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
    #         else self.init_config['obj_init_pos']
    #     print(f'Second init pos: {self.obj_init_pos}.')
    #     self.sim.model.body_pos[self.model.body_name2id(
    #         'coffee_machine'
    #     )] = self.obj_init_pos

    #     pos_mug = self.obj_init_pos + np.array([.0, -.22, .0])
    #     self._set_obj_xyz(pos_mug)

    #     pos_button = self.obj_init_pos + np.array([.0, -.22, .3])
    #     self._target_pos = pos_button + np.array([.0, self.max_dist, .0])

    #     return self._get_obs()

    def reset_model(self):
        self._reset_hand()
        self._random_init_color()

        print('Running reset.')
        print(f'First init pos: {self.obj_init_pos}.')
        if self.random_init:
            # obj_init_pos = self.obj_init_pos
            # obj_init_pos = np.array([0.5, 0.85, 0.0])
            self.obj_init_pos = self.random_init_coffee_machine_position()
        else:
            self.obj_init_pos = self.init_config['obj_init_pos']
        print(f'Second init pos: {self.obj_init_pos}.')
        self.sim.model.body_pos[self.model.body_name2id(
            'coffee_machine'
        )] = self.obj_init_pos
        quat = self._random_init_quat()

        if all(quat == QUAT_LIST[0]):
            pos_mug = self.obj_init_pos + np.array([.0, -.22, .0])
            pos_button = self.obj_init_pos + np.array([.0, -.22, .3])
            self._target_pos = pos_button + np.array([.0, self.max_dist, .0])
        elif all(quat == QUAT_LIST[1]):
            pos_mug = self.obj_init_pos + np.array([.22, .0, .0])
            pos_button = self.obj_init_pos + np.array([.22, .0, .3])
            self._target_pos = pos_button + np.array([-self.max_dist, .0, .0])
        elif all(quat == QUAT_LIST[2]):
            pos_mug = self.obj_init_pos + np.array([-.22, .0, .0])
            pos_button = self.obj_init_pos + np.array([-.22, .0, .3])
            self._target_pos = pos_button + np.array([self.max_dist, .0, .0])
        self._set_obj_xyz(pos_mug)

        # pos_button = self.obj_init_pos + np.array([.0, -.22, .3])
        # self._target_pos = pos_button + np.array([.0, self.max_dist, .0])
        self.num_resets += 1
        return self._get_obs()

    def random_init_coffee_machine_position(self):
        obj_init_pos = np.random.uniform((-0.5, 0.60, 0), (0.5, 0.85, 0))
        if obj_init_pos[1] < 0.70 and -0.2 < obj_init_pos[0] < 0.2:
            obj_init_pos = self.random_init_coffee_machine_position()
        if self.num_resets:
            print(f'Environment has been reset for '
                  f'{self.num_resets} times, break.')
            return self.obj_init_pos
        return obj_init_pos

    def _random_init_quat(self, index=None):
        if self.obj_init_pos[0] > 0:
            init_quat_list = [QUAT_LIST[0], QUAT_LIST[2]]
        else:
            init_quat_list = [QUAT_LIST[0], QUAT_LIST[1]]
        # init_quat_list = QUAT_LIST[:2]
        if index is None:
            index = random.randint(0, len(init_quat_list)-1)
        self.quat_index = index
        quat = np.array(init_quat_list[index])
        self.sim.model.body_quat[
            self.sim.model.body_name2id('coffee_machine')] = quat
        self.quat = quat
        return quat

    def _random_init_color(self):
        # rgb = random.choice(RGB_COLOR_LIST)
        # rgba = np.array(list(rgb) + [1.])

        def get_random_rgba():
            return np.array(list(random.choice(RGB_COLOR_LIST)) + [1.])

        def set_model_rgba(model_name: str):
            if model_name == 'coffee_machine_body':
                model_name = ['coffee_machine_body1', 'coffee_machine_body2']
            else:
                model_name = [model_name]
            rgba = get_random_rgba()
            for name in model_name:
                self.sim.model.geom_rgba[
                    self.sim.model.geom_name2id(name)] = rgba

        for model_name in ['coffee_machine_body', 'mug', 'handle']:
            set_model_rgba(model_name)

    def compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        if all(self.quat == QUAT_LIST[0]):
            obj_to_target = max(abs(self._target_pos[0] - obj[0]),
                                abs(self._target_pos[1] - obj[1]))
            # obj_to_target = abs(self._target_pos[1] - obj[1])
        elif all(self.quat == QUAT_LIST[1]) or all(self.quat == QUAT_LIST[2]):
            obj_to_target = max(abs(self._target_pos[0] - obj[0]),
                                abs(self._target_pos[1] - obj[1]))
            # obj_to_target = abs(self._target_pos[0] - obj[0])
        else:
            raise ValueError(f'Got unrecognizable quat: {self.quat}.')

        tcp_closed = max(obs[3], 0.0)
        near_button = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.05),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        button_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self.max_dist,
            sigmoid='long_tail',
        )

        reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.05:
            reward += 8 * button_pressed
        success = obj_to_target <= 0.01
        if success:
            reward = 10

        return (
            reward,
            tcp_to_obj,
            obs[3],
            obj_to_target,
            near_button,
            button_pressed,
            success
        )
