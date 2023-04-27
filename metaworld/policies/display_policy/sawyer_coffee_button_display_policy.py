import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.envs.display_utils import QUAT_LIST


class SawyerCoffeeButtonV2DisplayPolicy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'button_pos': obs[4:7],
            'button_quat': obs[7: 11],
            'unused_info': obs[11:],
        }

    def get_action(self, obs, info):
        o_d = self._parse_obs(obs)
        success = info.get('success', False)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d, success), p=10.)
        action['grab_effort'] = -1.

        return action.array

    @staticmethod
    def _desired_pos(o_d, success=False):
        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_pos']
        pos_button += 0 if success else np.array([.0, .0, -.07])
        button_quat = o_d['button_quat']

        print(f'Current position: {pos_curr}\n Button position: {pos_button}.')
        if np.linalg.norm(pos_curr - pos_button) > 0.1:
            # return np.array([pos_button[0], pos_curr[1], pos_button[2]])
            if all(button_quat == QUAT_LIST[0]):
                pos = pos_button + np.array([.0, -.1, .0])
                # print(f'无旋转: {pos}.')
                return pos
            elif all(button_quat == QUAT_LIST[1]):
                pos = pos_button + np.array([.1, .0, .0])
                # print(f'顺时针旋转: {pos}.')
                return pos
            elif all(button_quat == QUAT_LIST[2]):
                pos = pos_button + np.array([-.1, .0, .0])
                # print(f'逆时针旋转: {pos}.')
                return pos
        else:
            return pos_button
