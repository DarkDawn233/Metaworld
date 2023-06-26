import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.envs.display_utils import QUAT_LIST
from metaworld.envs.display_utils import safe_move

class SawyerDrawerCloseV2DisplayPolicy(Policy):

    def __init__(self, safe_move=False):
        self.safe_move = safe_move
        self.flag = False

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_grasp_info': obs[3],
            'drwr_pos': obs[4:7],
            'drwr_quat': obs[7:11],
            'unused_info': obs[11:],
        }

    def get_action(self, obs, info={}):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        if info.get('success', False):
            target_pos = [o_d['hand_pos'][0], o_d['hand_pos'][1], 0.3]
            action['delta_pos'] = move(o_d['hand_pos'], to_xyz=target_pos, p=25.)
            action['grab_effort'] = -1.
            return action.array

        if self.safe_move:
            desired_pos, self.flag = self._desired_pos(o_d, self.flag)
            if not self.flag:
                action['delta_pos'] = safe_move(o_d['hand_pos'], to_xyz=desired_pos, p=10.)
            else:
                action['delta_pos'] = move(o_d['hand_pos'], to_xyz=desired_pos, p=10.)
        else:
            desired_pos, self.flag = self._desired_pos(o_d, self.flag)
            action['delta_pos'] = move(o_d['hand_pos'], to_xyz=desired_pos, p=10.)
        action['grab_effort'] = 1.

        return action.array


    @staticmethod
    def _desired_pos(o_d, flag):
        pos_curr = o_d['hand_pos']
        pos_drwr = o_d['drwr_pos'] + np.array([.0, .0, -.02])

        quat = o_d['drwr_quat']
        quat_list = QUAT_LIST[:4]
        for i in range(len(quat_list)):
            if np.sum((quat - np.array(quat_list[i])) ** 2) < 1e-5:
                # print("index:", i)
                quat_index = i
                break

        if quat_index == 0:
            # if further forward than the drawer...
            if not flag:
                if pos_curr[2] < pos_drwr[2] + 0.23:
                    # rise up quickly (Z direction)
                    return np.array([pos_curr[0], pos_curr[1], pos_drwr[2] + 0.5]), False
                elif np.linalg.norm(pos_curr[:2] - (pos_drwr + np.array([0., -0.08, 0.23]))[:2]) > 0.02:
                    # move to front edge of drawer handle, but stay high in Z
                    return pos_drwr + np.array([0., -0.08, 0.23]), False
            # drop down to touch drawer handle
            if abs(pos_curr[2] - pos_drwr[2]) > 0.01:
                return pos_drwr + np.array([0., -0.08, 0.]), True
            # push toward drawer handle's centroid
            else:
                return pos_drwr, True
        elif quat_index == 3:
            if not flag:
                if pos_curr[2] < pos_drwr[2] + 0.23:
                    return np.array([pos_curr[0], pos_curr[1], pos_drwr[2] + 0.5]), False
                elif np.linalg.norm(pos_curr[:2] - (pos_drwr + np.array([0., +0.08, 0.23]))[:2]) > 0.03:
                    return pos_drwr + np.array([0., +0.08, 0.23]), False
            if abs(pos_curr[2] - pos_drwr[2]) > 0.01:
                return pos_drwr + np.array([0., +0.08, 0.]), True
            else:
                return pos_drwr, True
        elif quat_index == 1:
            if not flag:
                if pos_curr[2] < pos_drwr[2] + 0.23:
                    return np.array([pos_curr[0], pos_curr[1], pos_drwr[2] + 0.5]), False
                elif np.linalg.norm(pos_curr[:2] - (pos_drwr + np.array([+0.08, 0., 0.23]))[:2]) > 0.03:
                    return pos_drwr + np.array([+0.08, 0., 0.23]), False
            if abs(pos_curr[2] - pos_drwr[2]) > 0.01:
                return pos_drwr + np.array([+0.08, 0., 0.]), True
            else:
                return pos_drwr, True
        else:
            if not flag:
                if pos_curr[2] < pos_drwr[2] + 0.23:
                    return np.array([pos_curr[0], pos_curr[1], pos_drwr[2] + 0.5]), False
                elif np.linalg.norm(pos_curr[:2] - (pos_drwr + np.array([-0.08, 0., 0.23]))[:2]) > 0.03:
                    return pos_drwr + np.array([-0.08, 0., 0.23]), False
            if abs(pos_curr[2] - pos_drwr[2]) > 0.01:
                return pos_drwr + np.array([-0.08, 0., 0.]), True
            else:
                return pos_drwr, True