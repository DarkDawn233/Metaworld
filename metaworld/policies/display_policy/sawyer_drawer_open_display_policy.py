import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.envs.display_utils import QUAT_LIST

class SawyerDrawerOpenV2DisplayPolicy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
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

        # NOTE this policy looks different from the others because it must
        # modify its p constant part-way through the task
        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = -1.

        return action.array
                    
    @staticmethod
    def _desired_pos(o_d):
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
            # align end effector's Z axis with drawer handle's Z axis
            if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.03:
                return pos_drwr + np.array([0., 0., 0.3])
            # drop down to touch drawer handle
            elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
                return pos_drwr
            # push toward a point just behind the drawer handle
            # also increase p value to apply more force
            else:
                return pos_drwr + np.array([0., -0.08, 0.])
        elif quat_index == 3:
            if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.03:
                return pos_drwr + np.array([0., 0., 0.3])
            elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
                return pos_drwr
            else:
                return pos_drwr + np.array([0., +0.08, 0.])
        elif quat_index == 1:
            if pos_drwr[1] < 0.5:
                target_pos_drwr = pos_drwr + np.array([-0.04, +0.03, 0])
            else:
                target_pos_drwr = pos_drwr + np.array([-0.04, -0.03, 0])
            if np.linalg.norm(pos_curr[:2] - target_pos_drwr[:2]) > 0.03:
                return target_pos_drwr + np.array([0., 0., 0.3])
            elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
                return target_pos_drwr
            else:
                return target_pos_drwr + np.array([+0.08, 0., 0.])
        else:
            if pos_drwr[1] < 0.5:
                target_pos_drwr = pos_drwr + np.array([+0.04, +0.03, 0])
            else:
                target_pos_drwr = pos_drwr + np.array([+0.04, -0.03, 0])
            if np.linalg.norm(pos_curr[:2] - target_pos_drwr[:2]) > 0.03:
                return target_pos_drwr + np.array([0., 0., 0.3])
            elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
                return target_pos_drwr
            else:
                return target_pos_drwr + np.array([-0.08, 0., 0.])