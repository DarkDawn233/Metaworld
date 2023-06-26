import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.envs.display_utils import QUAT_LIST
from metaworld.envs.display_utils import safe_move

class SawyerCoffeeButtonV2DisplayPolicy(Policy):

    def __init__(self, safe_move=False):
        self.safe_move = safe_move
        self.flag = False

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

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        if info.get('success', False):
            button_quat = o_d['button_quat']
            pos_button = o_d['button_pos']
            pos_button += np.array([.0, .0, .1])
            if all(button_quat == QUAT_LIST[0]):
                target_pos = pos_button + np.array([.0, -.1, .0])
            elif all(button_quat == QUAT_LIST[1]):
                target_pos = pos_button + np.array([.1, .0, .0])
            elif all(button_quat == QUAT_LIST[2]):
                target_pos = pos_button + np.array([-.1, .0, .0])
            else:
                raise NotImplementedError(f"coffee machine quat 3 not implemented")
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
        action['grab_effort'] = -1.

        return action.array

    @staticmethod
    def _desired_pos(o_d, flag):
        # success = info.get('success', False)
        # pos_targ = info.get('target_pos',None)
        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_pos']
        pos_button += np.array([.0, .0, -.07])
        button_quat = o_d['button_quat']

        # print(f'Current position: {pos_curr}\n Button position: {pos_button}\n Target position: {pos_targ}')
        # if success:
        #     if all(button_quat == QUAT_LIST[0]):
        #         pos = pos_button + np.array([.0, -.1, .0])
        #         # print(f'无旋转: {pos}.')
        #         return pos
        #     elif all(button_quat == QUAT_LIST[1]):
        #         pos = pos_button + np.array([.1, .0, .0])
        #         # print(f'顺时针旋转: {pos}.')
        #         return pos
        #     elif all(button_quat == QUAT_LIST[2]):
        #         pos = pos_button + np.array([-.1, .0, .0])
        #         # print(f'逆时针旋转: {pos}.')
        #         return pos
        if not flag and np.linalg.norm(pos_curr - pos_button) > 0.1:
            # return np.array([pos_button[0], pos_curr[1], pos_button[2]])
            if all(button_quat == QUAT_LIST[0]):
                pos = pos_button + np.array([.0, -.1, .0])
                # print(f'无旋转: {pos}.')
            elif all(button_quat == QUAT_LIST[1]):
                pos = pos_button + np.array([.1, .0, .0])
                # print(f'顺时针旋转: {pos}.')
            elif all(button_quat == QUAT_LIST[2]):
                pos = pos_button + np.array([-.1, .0, .0])
                # print(f'逆时针旋转: {pos}.')
            else:
                raise NotImplementedError(f"coffee machine quat 3 not implemented")
            return pos, False
        else:
            return pos_button, True
