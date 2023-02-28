import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPegUnplugSideV2Policy(Policy):

    def __init__(self):
        self.grab_done = False
        self.t = 0
    
    def reset(self):
        self.grab_done = False
        self.t = 0

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_gripper': obs[3],
            'peg_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d, self.grab_done, self.t), p=25.)
        action['grab_effort'], self.grab_done = self._grab_effort(o_d, self.grab_done)
        if self.grab_done:
            self.t += 1

        return action.array

    @staticmethod
    def _desired_pos(o_d, grab_done, t):
        pos_curr = o_d['hand_pos']
        pos_peg = o_d['peg_pos'] + np.array([-.02, .0, .035])

        if grab_done and t < 15:
            return np.array([*pos_peg[:2], .15])
        elif grab_done:
            return pos_curr + np.array([.01, .0, .0])
        if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > 0.04:
            return pos_peg + np.array([0., 0., 0.2])
        elif abs(pos_curr[2] - 0.15) > 0.02:
            return np.array([*pos_peg[:2], .15])
            # return pos_peg
        else:
            return pos_curr + np.array([.01, .0, .0])

    @staticmethod
    def _grab_effort(o_d, grab_done):
        pos_curr = o_d['hand_pos']
        pos_peg = o_d['peg_pos'] + np.array([-.02, .0, .035])
        # print(pos_peg)

        if grab_done:
            return .6, True
        if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > 0.04 \
            or pos_curr[2] - 0.15 > 0.02:
            return -1., False
        else:
            return .6, True
