
from copy import deepcopy
import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.envs.display_utils import obstacle_in_path, near_obstacle


class SawyerDeskPlaceV2DisplayPolicy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'mug_pos': obs[4:7],
            # 'goal_xy': obs[-3:-1],
            'target_pos': obs[-3:],
            'unused_info_1': obs[7:-3],
            # 'unused_info_2': obs[-1],
        }

    def get_action(self, obs, info):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        desired_pos = self._desired_pos(o_d, info)

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=desired_pos, p=10.)

        action['grab_effort'] = self._grab_effort(o_d)
        print(f'Grab effort: {action["grab_effort"]}')
        if info.get('success', False):
            action['grab_effort'] = -1
            if not hasattr(self, 'success_halt_counts'):
                self.success_halt_counts = 0
            if self.success_halt_counts < 3:
                print('Halting after success.')
                action['delta_pos'] *= 0
                self.success_halt_counts += 1
        if not hasattr(self, 'near_cup_halt_cnts'):
            self.near_cup_halt_cnts = 0
        if (action['grab_effort'] > 0) and (self.near_cup_halt_cnts < 10):
            action['delta_pos'] *= 0
            self.near_cup_halt_cnts += 1
            print(f'Pause hand count: {self.near_cup_halt_cnts}')

        return action.array

    # @staticmethod
    # def _desired_pos(o_d, info):
    #     pos_curr = o_d['hand_pos']
    #     pos_mug = o_d['mug_pos'] + np.array([.01, .0, .05])
    #     pos_goal = o_d['goal_xy']

    #     if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.06:
    #         return pos_mug + np.array([.0, .0, .2])
    #     elif abs(pos_curr[2] - pos_mug[2]) > 0.02:
    #         return pos_mug
    #     else:
    #         return np.array([pos_goal[0]-pos_mug[0]+pos_curr[0],
    #                          pos_goal[1]-pos_mug[1]+pos_curr[1],
    #                          .1])

    @staticmethod
    def _desired_pos(o_d, info=None):
        if info is None:
            info = {}

        pos_curr = o_d['hand_pos']
        pos_targ = o_d['target_pos']
        pos_mug = o_d['mug_pos']
        pos_obj = info.get('obj_pos', pos_curr)
        pos_init = info.get('init_pos', np.array([0, 0, 0]))

        # if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.05:
        #     state = 'Approaching cup'
        #     pos_targ = deepcopy(pos_mug)
        #     pos_targ[2] = 0.42
        success = info.get('success', False)
        if success:
            state = 'Leaving cup.'
            pos_targ = np.array([pos_targ[0]-pos_mug[0]+pos_curr[0],
                                 pos_targ[1]-pos_mug[1]+pos_curr[1],
                                 .4])
            # if all(quat == QUAT_LIST[0]):
            #     pos_target += np.array([0.2, 0.0, 0.0])
            # elif all(quat == QUAT_LIST[1]):
            #     pos_target += np.array([0.2, 0.0, 0.0])
            # elif all(quat == QUAT_LIST[2]):
            #     pos_target += np.array([-0.2, 0.0, 0.0])
        elif pos_curr[2] - pos_mug[2] > 0.080:
            state = 'Picking cup'
            pos_targ = deepcopy(pos_mug)
        elif np.linalg.norm(pos_curr[:2] - pos_init[:2]) < 0.02 \
                and pos_curr[2] < 0.40:
            state = 'Lifting up mug cup.'
            pos_targ = deepcopy(pos_curr)
            pos_targ[2] = 0.45
        # elif np.linalg.norm(pos_curr[:2] - pos_targ[:2]) > 0.05:
        #     state = 'Moving mug cup.'
        #     pos_targ = deepcopy(pos_targ)
        #     pos_targ[2] = 0.45
        elif obstacle_in_path(pos_curr, pos_targ, pos_obj) \
                or near_obstacle(pos_curr, pos_obj):
            state = f'>>> Leaving obstacle, target position: {pos_targ}'
            if np.linalg.norm(pos_curr[:2] - pos_targ[:2]) > 0.02:
                state = 'Moving mug cup.'
                pos_targ = deepcopy(pos_targ)
                pos_targ[2] = 0.45
        else:
            state = '**** Near target ****'
            pos_targ = deepcopy(pos_targ)
            pos_targ[2] += (pos_curr[2] - pos_mug[2])
        # if (abs(pos_curr[0] - pos_targ[0]) > abs(pos_obj[0] - pos_targ[0])) \
        #         and ((pos_curr[0] - pos_targ[0]) * (pos_obj[0] - pos_targ[0]) > 0):
        #     state = f'>>> Leaving obstacle, target position: {pos_targ}'
        #     # pos_targ = deepcopy(pos_curr)
        #     # pos_targ[2] = 0.45
        #     if np.linalg.norm(pos_curr[:2] - pos_targ[:2]) > 0.02:
        #         state = 'Moving mug cup.'
        #         pos_targ = deepcopy(pos_targ)
        #         pos_targ[2] = 0.45
        #     # if pos_curr[2] < 0.40:
        #     #     # pos_targ[:2] = pos_curr[:2]
        #     #     pos_targ = deepcopy(pos_curr)
        #     #     pos_targ[2] = 0.45

        print(state)
        print(f'Current Position: {pos_curr}\n Target Position: {pos_targ}\n'
              f' Object Position: {pos_obj}\nMug cup position: {pos_mug}')

        return pos_targ

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_mug = o_d['mug_pos'] + np.array([.01, .0, .05])

        if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.05 or \
                abs(pos_curr[2] - pos_mug[2]) > 0.080:
            return -1.
        else:
            return .5