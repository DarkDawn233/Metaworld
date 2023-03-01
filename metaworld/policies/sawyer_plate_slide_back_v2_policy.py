import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPlateSlideBackV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_1': obs[3],
            'puck_pos': obs[4:7],
            'unused_2': obs[7:-3],
            'goal_pos': obs[-3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = -1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck_ = o_d['puck_pos'] + np.array([.0, -.065, .025])
        pos_puck = o_d['puck_pos']
        pos_goal = o_d['goal_pos']

        if np.linalg.norm(pos_curr[:2] - pos_puck_[:2]) > 0.01:
            return pos_puck_ + np.array([.0, .0, .1])
        elif abs(pos_curr[2] - pos_puck_[2]) > 0.04:
            return pos_puck_
        elif pos_curr[1] > .7:
            return pos_curr + np.array([.0, -.1, .0])
        elif pos_curr[1] > .6:
            return np.array([.15, .55, pos_curr[2]])
        else:
            return np.array([pos_goal[0] - pos_puck[0] + pos_curr[0], pos_goal[1] - pos_puck[1] + pos_curr[1], pos_curr[2]])
            # return np.array([pos_curr[0] - .1, .55, pos_curr[2]])
