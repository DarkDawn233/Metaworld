import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPushV2Policy(Policy):
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
            'puck_pos': obs[4:7],
            'unused_2':  obs[7:-3],
            'goal_pos': obs[-3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d, self.grab_done, self.t), p=10.)
        action['grab_effort'], self.grab_done = self._grab_effort(o_d, self.grab_done)
        if self.grab_done:
            self.t += 1

        return action.array

    @staticmethod
    def _desired_pos(o_d, grab_done, t):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos']
        pos_goal = o_d['goal_pos']

        if grab_done and t > 20:
            return pos_goal - pos_puck + pos_curr
        elif grab_done:
            return pos_curr
        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02:
            return pos_puck + np.array([0., 0., 0.2])
        # Once XY error is low enough, drop end effector down on top of puck
        elif abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return pos_puck + np.array([0., 0., 0.03])
        # Move to the goal
        else:
            return pos_goal - pos_puck + pos_curr

    @staticmethod
    def _grab_effort(o_d, grab_done):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos']

        if grab_done:
            return 0.6, True
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02 or abs(pos_curr[2] - pos_puck[2]) > 0.10:
            return 0., False
        # While end effector is moving down toward the puck, begin closing the grabber
        else:
            return 0.6, True
