import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerShelfPlaceV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_1': obs[3],
            'block_pos': obs[4:7],
            'unused_2':  obs[7:-3],
            'shelf_x': obs[-3],
            'shelf_yz': obs[-2:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_block = o_d['block_pos'] + np.array([-.005, .0, .015])
        pos_block_ = o_d['block_pos']
        pos_shelf_x = o_d['shelf_x']
        pos_shelf = np.array([o_d['shelf_x'], o_d['shelf_yz'][0], o_d['shelf_yz'][1]])

        if np.linalg.norm(pos_curr[:2] - pos_block[:2]) > 0.04:
            # positioning over block
            return pos_block + np.array([0., 0., 0.3])
        elif abs(pos_curr[2] - pos_block[2]) > 0.04:
            # grabbing block
            return pos_block
        elif np.abs(pos_curr[0] - pos_shelf_x) > 0.02:
            # centering with goal pos
            return np.array([pos_shelf_x, pos_curr[1], 0.3])
        elif pos_curr[2] < 0.30:
            # move up to correct height
            pos_new = pos_curr + np.array([0., 0., 0.30])
            return pos_new
        else:
            # move forward to goal
            return pos_shelf - pos_block_ + pos_curr
            # pos_new = pos_curr + np.array([0., 0.05, 0.])
            # return pos_new

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_block = o_d['block_pos']

        if np.linalg.norm(pos_curr[:2] - pos_block[:2]) > 0.04 \
            or abs(pos_curr[2] - pos_block[2]) > 0.15:
            return -1.
        else:
            return .7
