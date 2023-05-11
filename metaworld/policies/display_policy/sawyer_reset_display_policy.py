from copy import deepcopy
import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.envs.display_utils import obstacle_in_path, near_obstacle


class SawyerResetV2DisplayPolicy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'button_pos': obs[4:7],
            'button_quat': obs[7: 11],
            'target_pos': obs[-3:],
            'unused_info': obs[11:-3],
        }

    def get_action(self, obs, info):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'],
                                   to_xyz=self._desired_pos(o_d, info),
                                   p=10.)
        action['grab_effort'] = -1.

        return action.array

    @staticmethod
    def _desired_pos(o_d, info=None):
        pos_curr = o_d['hand_pos']
        pos_targ = o_d['target_pos']
        pos_obj = info.get('obj_pos', pos_curr)
        # if abs(pos_curr[0] - pos_targ[0]) > abs(pos_obj[0] - pos_targ[0]):
        #     pos_targ = np.copy(pos_obj)
        #     if pos_obj[1] < 0.6:
        #         pos_targ[:2] += 0.15
        #     else:
        #         pos_targ[:2] -= 0.15
        #     pos_targ[2] = 0.45
        if np.linalg.norm(pos_curr[:2] - pos_targ[:2]) > 0.1:
            pos_targ = deepcopy(pos_targ)
            pos_targ[2] = 0.45
        if obstacle_in_path(pos_curr, pos_targ, pos_obj) \
                or near_obstacle(pos_curr, pos_obj):
            if pos_curr[2] < 0.4:
                pos_targ[:2] = pos_curr[:2]
        print(f'Current Position: {pos_curr}\n Target Position: {pos_targ}\n'
              f' Object Position: {pos_obj}')
        return pos_targ
