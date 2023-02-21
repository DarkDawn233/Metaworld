import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerBoxCloseV2Policy(Policy):

    def __init__(self):
        self.grab_done = False

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'lid_pos': obs[4:7],
            'extra_info_1': obs[7:-3],
            'box_pos': obs[-3:-1],
            'extra_info_2': obs[-1],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d, self.grab_done), p=25.)
        action['grab_effort'], self.grab_done = self._grab_effort(o_d, self.grab_done)

        return action.array
    
    def reset(self):
        self.grab_done = False

    @staticmethod
    def _desired_pos(o_d, grab_done):
        pos_curr = o_d['hand_pos']
        pos_lid = o_d['lid_pos'] + np.array([.0, .0, +.02])
        # pos_box = np.array([*o_d['box_pos'], 0.15]) + np.array([.0, .0, .0])
        pos_box = np.array([*o_d['box_pos'], 0.096]) + np.array([.0, .0, .0])
        # print("pos_curr:", pos_curr)
        # print("pos_lid:", pos_lid)
        # print("pos_box:", pos_box)

        if grab_done:
            if abs(pos_curr[2] - pos_lid[2]) > 0.05:
                return pos_lid
            elif np.linalg.norm(pos_lid[:2] - pos_box[:2]) < 0.01:
                # return np.array([pos_box[0]+pos_curr[0]-pos_lid[0], pos_box[1]+pos_curr[1]-pos_lid[1], pos_box[2]])
                return np.array([pos_curr[0], pos_curr[1], pos_box[2]])
            elif pos_lid[2] < pos_box[2] + 0.01:
                return np.array([pos_curr[0], pos_curr[1], 0.15 + pos_curr[2] - pos_lid[2]])
            # Move to the goal
            else:
                return np.array([pos_box[0]+pos_curr[0]-pos_lid[0], pos_box[1]+pos_curr[1]-pos_lid[1], pos_curr[2]])


        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01 and pos_curr[2] - pos_lid[2] < 0.1:
            return np.array([pos_curr[0], pos_curr[1], pos_lid[2]+0.1])
        if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01:
            return np.array([*pos_lid[:2], 0.2])
        # Once XY error is low enough, drop end effector down on top of puck
        elif abs(pos_curr[2] - pos_lid[2]) > 0.05:
            return pos_lid
        # If not at the same Z height as the goal, move up to that plane
        elif pos_curr[2] < 0.15 - 0.02:
            return np.array([pos_curr[0], pos_curr[1], 0.15])
        # Move to the goal
        else:
            return np.array([pos_box[0], pos_box[1], pos_curr[2]])

    @staticmethod
    def _grab_effort(o_d, grab_done):
        pos_curr = o_d['hand_pos']
        pos_lid = o_d['lid_pos'] + np.array([.0, .0, +.02])

        if grab_done:
            return 1., True
        elif np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01 or abs(pos_curr[2] - pos_lid[2]) > 0.05:
            return 0., False
        # While end effector is moving down toward the puck, begin closing the grabber
        else:
            return 1., True
