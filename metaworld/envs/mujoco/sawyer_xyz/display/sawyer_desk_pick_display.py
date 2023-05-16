import random
import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_display_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_base import SawyerXYZEnvDisplay
from metaworld.envs.display_utils import RGB_COLOR_LIST, QUAT_LIST, random_grid_pos


class SawyerDeskPickEnvV2Display(SawyerXYZEnvDisplay):

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.55, -.001)
        obj_high = (0.1, 0.65, +.001)
        goal_low = (-0.05, 0.7, -.001)
        goal_high = (0.05, 0.75, +.001)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0., .6, 0.]),
            'hand_init_pos': np.array([0., .4, .2]),
        }
        self.goal = np.array([0., 0.75, 0])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']
        self.coffee_machine_init_pos = np.array([0.0, 0.6, 0.0])

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self.num_resets = 0
        self.quat = QUAT_LIST[0]
        self.succeed = False

    @property
    def model_name(self):
        return full_display_path_for('sawyer_xyz/sawyer_coffee.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        reward, tcp_to_obj, tcp_open, obj_to_target, grasp_reward, in_place = self.compute_reward_desk_pick(
            action, obs)
        success = float(obj_to_target <= 0.02)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(self.touching_object and (tcp_open > 0))

        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
            # 'obj_pos': self.sim.model.body_pos[
            #     self.model.body_name2id('coffee_machine')],
            'obj_pos': self.coffee_machine_init_pos,
        }
        info['quat'] = self._get_quat_objects()
        # print(f'-- Object position: {info["obj_pos"]}')

        if success:
            self.succeed = True
            hand_pos = self.get_endeff_pos()
            info['after_success'] = info['success'] and (hand_pos[2] >= 0.39)
        if self.succeed:
            info['success'] = True
            hand_pos = self.get_endeff_pos()
            info['after_success'] = info['success'] and (hand_pos[2] >= 0.39)

        return reward, info

    # def step(self, a):
    #     obs, reward, done, info = super().step(a)
    #     info['quat'] = self.quat
    #     return obs, reward, done, info

    @property
    def _target_site_config(self):
        return [('coffee_goal', self._target_pos)]

    def _get_pos_coffee_machine(self):
        return self.get_body_com('coffee_machine')

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _get_quat_coffee_machine(self):
        return self.get_body_quat('coffee_machine')

    def _get_quat_objects(self):
        return self.get_body_quat('obj')
        # return Rotation.from_matrix(
        #     self.data.get_geom_xmat('mug')
        # ).as_quat()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        # addr_mug = self.model.get_joint_qpos_addr('mug_obj')
        # print("adr:", addr_mug)
        # print("qpos:", qpos)
        # print("qvel:", qvel)
        qpos[0:3] = pos.copy()
        # qpos[3:7] = np.array([1, 0, 0, 0])
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    # def reset_model(self):
    #     self._reset_hand()

    #     pos_mug_init = self.init_config['obj_init_pos']
    #     pos_mug_goal = self.goal

    #     if self.random_init:
    #         pos_mug_init, pos_mug_goal = np.split(self._get_state_rand_vec(), 2)
    #         while np.linalg.norm(pos_mug_init[:2] - pos_mug_goal[:2]) < 0.15:
    #             pos_mug_init, pos_mug_goal = np.split(
    #                 self._get_state_rand_vec(),
    #                 2
    #             )

    #     self._set_obj_xyz(pos_mug_init)
    #     self.obj_init_pos = pos_mug_init

    #     pos_machine = pos_mug_goal + np.array([.0, .22, .0])
    #     self.sim.model.body_pos[self.model.body_name2id(
    #         'coffee_machine'
    #     )] = pos_machine

    #     self._target_pos = pos_mug_goal
    #     return self._get_obs()

    def reset_model(self):
        print('===============')
        print('Running reset.')
        print('===============')
        self._reset_hand_init_pos()
        self._random_init_color()
        self._random_table_and_floor()
        self.random_reset_target()
        self.reset_objects()
        self._reset_hand()

        self.num_resets += 1
        print('Target position', self._target_pos)
        return self._get_obs()

    def _reset_hand_init_pos(self):
        # Init hand first will cause cup bugs.
        # To keep coffee machine far from the hand, we init the hand pos first.
        lower_bound = (-0.5, 0.40, 0.30)
        upper_bound = ( 0.5, 0.80, 0.35)
        self.hand_init_pos = np.random.uniform(lower_bound, upper_bound)

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
        print(f'Hand position randomly set as {self.hand_init_pos}.')

    def random_reset_target(self):
        lower_bound = (-0.5, 0.40, 0.40)
        upper_bound = ( 0.5, 0.80, 0.40)
        self._target_pos = np.random.uniform(lower_bound, upper_bound)

    def reset_objects(self):
        pos_mug_init = self.init_config['obj_init_pos']
        pos_mug_goal = self._target_pos

        if self.random_init and self.num_resets == 0:
            pos_mug_init = self.random_mug_cup_position()
            # pos_mug_goal = self.random_mug_cup_position()
            while np.linalg.norm(pos_mug_init[:2] - pos_mug_goal[:2]) < 0.1:
                pos_mug_init = self.random_mug_cup_position()
        else:
            pos_mug_init = self.obj_init_pos
            pos_mug_goal = self._target_pos
        pos_mug_init[2] = 0.
        # pos_mug_goal[2] = np.random.uniform(0.40, 0.45)

        self._set_obj_xyz(pos_mug_init)
        self.obj_init_pos = pos_mug_init
        self.goal = pos_mug_goal

        print(f'Cup init position: {self.obj_init_pos}')
        print(f'Cup goal position: {self.goal}')

        pos_machine = self.random_init_coffee_machine_position_and_quat()
        print(f'Coffee machine position: {pos_machine}.')
        print(f'Coffee machine rotation: {self.quat}.')

    def random_init_coffee_machine_position_and_quat(self, index=None):
        if self.num_resets:
            return self.coffee_machine_init_pos
        xrange = (-0.5, 0.5)
        yrange = (0.60, 0.8)
        radius = 0.25
        forbidden_region = (
            (
                (self.obj_init_pos[0]-radius, self.obj_init_pos[0]+radius),
                (self.obj_init_pos[1]-radius, self.obj_init_pos[1]+radius),
                # (0, 0),
            ),
            # (
            #     (self.hand_init_pos[0]-radius, self.hand_init_pos[0]+radius),
            #     # (self.hand_init_pos[1]-radius, self.hand_init_pos[1]+radius),
            #     (0, 0),
            # ),
            (
                (self._target_pos[0]-radius, self._target_pos[0]+radius),
                (self._target_pos[1]-radius, self._target_pos[1]+radius),
                # (0, 0),
            ),
        )
        pos_machine = random_grid_pos(xrange, yrange, forbidden_region)
        pos_machine = np.array([*pos_machine, 0])
        if pos_machine[0] > 0:
            init_quat_list = [QUAT_LIST[0], QUAT_LIST[2]]
        else:
            init_quat_list = [QUAT_LIST[0], QUAT_LIST[1]]
        if index is None:
            index = random.randint(0, len(init_quat_list)-1)
        self.quat_index = index
        quat = np.array(init_quat_list[index])
        self.sim.model.body_quat[
            self.sim.model.body_name2id('coffee_machine')] = quat
        # self.sim.model.body_quat[
        #     self.sim.model.body_name2id('mug')] = quat
        self.quat = quat
        # if all(quat == QUAT_LIST[0]):
        #     pos_machine = self.goal + np.array([.0, .22, .0])
        # elif all(quat == QUAT_LIST[1]):
        #     pos_machine = self.goal + np.array([-.22, .0, .0])
        # elif all(quat == QUAT_LIST[2]):
        #     pos_machine = self.goal + np.array([.22, .0, .0])
        self.sim.model.body_pos[self.model.body_name2id(
            'coffee_machine')] = pos_machine
        # print(f'\n\nInitialized: mug cup position: {self.obj_init_pos} '
        #       f'coffee machine position: {pos_machine} '
        #       f'quat index: {index} '
        #       f'coffee machine rotation: {quat}.\n\n')
        self.coffee_machine_init_pos = pos_machine
        return pos_machine

    def random_mug_cup_position(self):
        obj_init_pos: tuple = random_grid_pos(x_range=(-0.5, 0.5),
                                              y_range=(0.40, 0.80))
        obj_init_pos = np.array([*obj_init_pos, 0])
        # obj_init_pos = np.random.uniform((-0.36, 0.40, 0), (0.36, 0.60, 0))
        # if obj_init_pos[1] < 0.70 and -0.2 < obj_init_pos[0] < 0.2:
        #     obj_init_pos = self.random_mug_cup_position()
        if self.num_resets:
            print(f'Environment has been reset for '
                  f'{self.num_resets} times, break.')
            return self.obj_init_pos
        return obj_init_pos

    def _random_init_color(self):
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

    def compute_reward_desk_pick(self, action, obs):
        obj = obs[4:7]
        target = self._target_pos.copy()

        # Emphasize X and Y errors
        scale = np.array([2., 2., 1.])
        target_to_obj = (obj - target) * scale
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.obj_init_pos - target) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, 0.05),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )
        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.04,
            obj_radius=0.02,
            pad_success_thresh=0.05,
            xz_thresh=0.05,
            desired_gripper_effort=0.7,
            medium_density=True
        )

        reward = reward_utils.hamacher_product(object_grasped, in_place)

        if tcp_to_obj < 0.04 and tcp_opened > 0:
            reward += 1. + 5. * in_place
        if np.linalg.norm(obj - target) <= 0.02:
            reward = 10.
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            np.linalg.norm(obj - target),  # recompute to avoid `scale` above
            object_grasped,
            in_place
        )
