import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.envs.display_utils import safe_move

class SawyerPlaceV2DisplayPolicy(Policy):

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

        # print("pos_hand:", o_d['hand_pos'])
        # print("pos_mug:", o_d['mug_pos'])
        # print("grasp:", o_d['grasp_info'])
        # print("pos_goal:", o_d['goal_pos'])

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        if info.get('success', False) or self.success:
            self.success = True
            if info.get('task_name', None) == 'drawer-place':
                # print('grasp_info:', o_d['grasp_info'])
                if o_d['grasp_info'] < 0.7:
                    target_pos = o_d['hand_pos']
                    action['grab_effort'] = -0.5
                    action['delta_pos'] = move(o_d['hand_pos'], to_xyz=target_pos, p=5.)
                else:
                    target_pos = [o_d['hand_pos'][0], o_d['hand_pos'][1], 0.3]
                    # print('hand_pos:', o_d['hand_pos'])
                    if o_d['hand_pos'][-1] < 0.2:
                        action['grab_effort'] = 0.
                    else:
                        action['grab_effort'] = -1.
                    action['delta_pos'] = move(o_d['hand_pos'], to_xyz=target_pos, p=10.)
                
            else:
                if o_d['grasp_info'] < 0.9:
                    target_pos = o_d['hand_pos']
                    action['delta_pos'] = move(o_d['hand_pos'], to_xyz=target_pos, p=5.)
                else:
                    target_pos = [o_d['hand_pos'][0], o_d['hand_pos'][1], 0.3]
                    action['delta_pos'] = move(o_d['hand_pos'], to_xyz=target_pos, p=10.)
                action['grab_effort'] = -1.
            # return action.array
        else:
            to_pos, self.flag = self._desired_pos(o_d, self.flag, info)
            action['delta_pos'] = safe_move(o_d['hand_pos'], to_xyz=to_pos, p=10.)
            action['grab_effort'] = self._grab_effort(o_d, self.flag)

        return action.array


    @staticmethod
    def _desired_pos(o_d, flag, info):
        pos_curr = o_d['hand_pos']
        pos_mug = o_d['mug_pos'] + np.array([0., 0., 0.06])
        pos_goal = o_d['goal_pos']
        grasp_info = o_d['grasp_info']
        # gripper_separation = o_d['gripper_distance_apart']
        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if info.get('task_name', None) == 'drawer-place':
            pos_targ = pos_goal + np.array([0., 0., 0.09])
        else:
            pos_targ = pos_goal + np.array([0., 0., 0.06])

        # print(f'Current Position: {pos_curr}')
        if flag:
            # print(f' Target Position: {pos_targ}')
            return pos_targ, True
        if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) <= 0.02 and \
            abs(pos_curr[2] - pos_mug[2]) <= 0.01:
            if grasp_info > 0.6:
                # print(f' Target Position: {pos_mug}')
                return pos_mug, False
            else:
                # print(f' Target Position: {pos_targ}')
                return pos_targ, True
        else:
            # print(f' Target Position: {pos_mug}')
            return pos_mug, False

    
    @staticmethod
    def _grab_effort(o_d, flag):
        pos_curr = o_d['hand_pos']
        pos_mug = o_d['mug_pos'] + np.array([0., 0., 0.06])

        if flag or (np.linalg.norm(pos_curr[:2] - pos_mug[:2]) <= 0.02 and \
            abs(pos_curr[2] - pos_mug[2]) <= 0.01):
            return 1.
        else:
            return 0.