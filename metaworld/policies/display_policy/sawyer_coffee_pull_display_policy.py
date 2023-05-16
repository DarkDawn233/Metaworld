import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.envs.display_utils import RGB_COLOR_LIST, QUAT_LIST


class SawyerCoffeePullV2DisplayPolicy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'mug_pos': obs[4:7],
            'unused_info': obs[7:-3],
            'target_pos': obs[-3:]
        }

    def get_action(self, obs, info):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'],
                                   to_xyz=self._desired_pos(o_d, info),
                                   p=10.)
        action['grab_effort'] = self._grab_effort(o_d)

        # if info.get('success', False):
        #     action['grab_effort'] = -1

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
        if (action['grab_effort'] > 0) and (self.near_cup_halt_cnts < 20):
            action['delta_pos'] *= 0
            self.near_cup_halt_cnts += 1
            print(f'Near cup halt counts: {self.near_cup_halt_cnts}')
        return action.array

    @staticmethod
    def _desired_pos(o_d, info=None):
        if info is None:
            info = {}
        quat = info.get('quat', QUAT_LIST[0])
        quat = np.array(quat)
        pos_curr = o_d['hand_pos']
        pos_mug = o_d['mug_pos'] + np.array([-.000, .0, .05])

        success = info.get('success', False)
        if success:
            state = 'Leaving cup.'
            pos_target = o_d['target_pos']
            pos_target[2] = 0.4
        elif np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.1:
            state = 'Approaching mug cup.'
            if all(quat == QUAT_LIST[0]):
                pos_target = pos_mug + np.array([.0, -.10, .15])
            elif all(quat == QUAT_LIST[1]):
                pos_target = pos_mug + np.array([.10, .0, .15])
            elif all(quat == QUAT_LIST[2]):
                pos_target = pos_mug + np.array([-.10, .0, .15])
        elif abs(pos_curr[2] - pos_mug[2]) > 0.02:
            state = 'Pickup mug cup.'
            pos_target = pos_mug
        elif np.linalg.norm(pos_curr[:2] - o_d['target_pos'][:2]) > 0.05:
            state = 'Moving mug cup on air.'
            pos_target = o_d['target_pos'] + np.array([.0, .0, .30])
        else:
            state = 'Moving mug cup to target position.'
            pos_target = o_d['target_pos']
        # print(f'State: {state}')
        # print(f'Current position: {pos_curr}.')
        # print(f' Target position: {pos_target}.')
        # print(f'Mug cup position: {pos_mug}.')
        return pos_target

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_mug = o_d['mug_pos'] + np.array([.01, .0, .05])

        if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.05 or \
            abs(pos_curr[2] - pos_mug[2]) > 0.02:
            return -1.
        else:
            return .7
