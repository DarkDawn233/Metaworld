import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.envs.display_utils import QUAT_LIST

class SawyerDrawerPlaceV2DisplayPolicy(Policy):

    def __init__(self):
        self.flag = False
        self.success = False

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'grasp_info': obs[3],
            'mug_pos': obs[4:7],
            'mug_quat': obs[7:11],
            'unused_info': obs[11:-3],
            'goal_pos': obs[-3:]
        }

    def get_action(self, obs, info={}):
        o_d = self._parse_obs(obs)

        print("pos_hand:", o_d['hand_pos'])
        print("pos_mug:", o_d['mug_pos'])
        print("grasp:", o_d['grasp_info'])
        print("pos_goal:", o_d['goal_pos'])

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        if info.get('success', False) or self.success:
            self.success = True
            if o_d['grasp_info'] < 0.9:
                target_pos = o_d['hand_pos']
            else:
                target_pos = [o_d['hand_pos'][0], o_d['hand_pos'][1], 0.3]
            action['delta_pos'] = move(o_d['hand_pos'], to_xyz=target_pos, p=10.)
            action['grab_effort'] = -1.
            return action.array

        to_pos, self.flag = self._desired_pos(o_d, self.flag)
        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=to_pos, p=10.)
        action['grab_effort'] = self._grab_effort(o_d, self.flag)

        return action.array


    @staticmethod
    def _desired_pos(o_d, flag):
        pos_curr = o_d['hand_pos']
        pos_mug = o_d['mug_pos'] + np.array([0., 0., 0.08])
        pos_goal = o_d['goal_pos']
        grasp_info = o_d['grasp_info']
        # gripper_separation = o_d['gripper_distance_apart']
        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if flag:
            return pos_goal + np.array([0., 0., 0.12]), True
        if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.02:
            return pos_mug + np.array([0., 0., 0.3]), False
        elif grasp_info > 0.6:
            return pos_mug, False
        elif pos_curr[2] < 0.29:
            return pos_curr + np.array([0., 0., 0.3]), False
        elif np.linalg.norm(pos_curr[:2] - pos_goal[:2]) > 0.02:
            return pos_goal + np.array([0., 0., 0.3]), False
        else:
            return pos_goal + np.array([0., 0., 0.12]), True

    
    @staticmethod
    def _grab_effort(o_d, flag):
        pos_curr = o_d['hand_pos']
        pos_mug = o_d['mug_pos'] + np.array([0., 0., 0.08])
        # pos_goal = o_d['goal_pos']

        if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) <= 0.02 and \
            abs(pos_curr[2] - pos_mug[2]) <= 0.01:
            return 1.
        else:
            return 0.