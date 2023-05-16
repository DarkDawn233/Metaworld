import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.envs.display_utils import QUAT_LIST


class SawyerCoffeePushV2DisplayPolicy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'mug_pos': obs[4:7],
            'goal_xy': obs[-3:-1],
            'unused_info_1': obs[7:-3],
            'unused_info_2': obs[-1],
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
        quat = info.get('quat', QUAT_LIST[0])
        quat = np.array(quat)
        pos_curr = o_d['hand_pos']
        pos_mug = o_d['mug_pos'] + np.array([.00, .0, .05])
        pos_goal = o_d['goal_xy']

        success = info.get('success', False)
        if success:
            state = 'Leaving cup.'
            pos_target = np.array([pos_goal[0]-pos_mug[0]+pos_curr[0],
                                   pos_goal[1]-pos_mug[1]+pos_curr[1],
                                   .4])
            if all(quat == QUAT_LIST[0]):
                pos_target += np.array([0.0, 0.0, 0.0])
            elif all(quat == QUAT_LIST[1]):
                pos_target += np.array([0.0, 0.0, 0.0])
            elif all(quat == QUAT_LIST[2]):
                pos_target += np.array([-0.0, 0.0, 0.0])
        elif np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.02:
            state = 'Approaching mug cup.'
            pos_target = pos_mug + np.array([.0, .0, .2])
        elif abs(pos_curr[2] - pos_mug[2]) > 0.03:
            state = 'Picking up mug cup.'
            pos_target = pos_mug
        elif np.linalg.norm(pos_curr[:2] - pos_goal) > 0.1:
            if pos_curr[2] < 0.25:
                state = '>>>> Moving mug cup on air 1.'
                pos_target = pos_mug
                pos_target[2] = 0.3
            else:
                state = '>>>> Moving mug cup on air 2.'
                pos_target = np.array([pos_goal[0]-pos_mug[0]+pos_curr[0],
                                       pos_goal[1]-pos_mug[1]+pos_curr[1],
                                       .3])
                offset = 0.08
                if all(quat == QUAT_LIST[0]):
                    pos_target = pos_target + np.array([.0, -offset, .0])
                if all(quat == QUAT_LIST[1]):
                    pos_target = pos_target + np.array([offset, .0, .0])
                if all(quat == QUAT_LIST[2]):
                    pos_target = pos_target + np.array([-offset, .0, .0])
        elif np.linalg.norm(pos_curr[:2] - pos_goal) > 0.02:
            state = 'Approaching target position.'
            pos_target = np.array([pos_goal[0]-pos_mug[0]+pos_curr[0],
                                   pos_goal[1]-pos_mug[1]+pos_curr[1],
                                   .00])
        else:
            state = 'Placing mug cup to target position.'
            pos_target = np.array([pos_goal[0]-pos_mug[0]+pos_curr[0],
                                   pos_goal[1]-pos_mug[1]+pos_curr[1],
                                   .00])
        # print(f'State: {state}')
        # print(f'Goal xy: {pos_goal}')
        # print(f'Current position: {pos_curr}.')
        # print(f' Target position: {pos_target}.')
        return pos_target

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_mug = o_d['mug_pos'] + np.array([.01, .0, .05])

        if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.02 or \
                abs(pos_curr[2] - pos_mug[2]) > 0.04:
            return -1.
        else:
            return .5
