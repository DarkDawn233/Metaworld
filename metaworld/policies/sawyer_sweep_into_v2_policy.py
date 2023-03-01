import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerSweepIntoV2Policy(Policy):

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
            'unused_1': obs[3],
            'cube_pos': obs[4:7],
            'unused_2':  obs[7:-3],
            'goal_pos': obs[-3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d, self.grab_done, self.t), p=25.)
        action['grab_effort'], self.grab_done = self._grab_effort(o_d, self.grab_done)
        self.t += self.grab_done

        return action.array

    @staticmethod
    def _desired_pos(o_d, grab_done, t):
        pos_curr = o_d['hand_pos']
        pos_cube = o_d['cube_pos'] + np.array([-0.005, .0, .01])
        pos_goal = o_d['goal_pos']

        if grab_done and t < 20:
            return pos_curr
        elif grab_done:
            if np.linalg.norm(pos_goal[:2] - pos_cube[:2]) > 0.04:
                return [pos_goal[0] - pos_cube[0] + pos_curr[0], pos_goal[1] - pos_cube[1] + pos_curr[1], pos_curr[2]]
            else:
                return pos_goal - pos_cube + pos_curr

        if np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > 0.04:
            return pos_cube + np.array([0., 0., 0.3])
        elif abs(pos_curr[2] - pos_cube[2]) > 0.04:
            return pos_cube
        elif np.linalg.norm(pos_goal[:2] - pos_cube[:2]) > 0.04:
            return [pos_goal[0] - pos_cube[0] + pos_curr[0], pos_goal[1] - pos_cube[1] + pos_curr[1], pos_curr[2]]
        else:
            return pos_goal - pos_cube + pos_curr

    @staticmethod
    def _grab_effort(o_d, grab_done):
        pos_curr = o_d['hand_pos']
        pos_cube = o_d['cube_pos']

        if grab_done:
            return .7, True
        if np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > 0.04 \
            or abs(pos_curr[2] - pos_cube[2]) > 0.04:
            return -1., False
        else:
            return .7, True
