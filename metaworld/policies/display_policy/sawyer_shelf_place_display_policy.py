import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerShelfPlaceV2DisplayPolicy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            "hand_pos": obs[:3],
            "unused_1": obs[3],
            "block_pos": obs[4:7],
            "unused_2": obs[7:-3],
            "shelf_x": obs[-3],
            "shelf_yz": obs[-2:]
        }

    def get_action(self, obs, info={}):
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d, info), p=25.0
        )
        action["grab_effort"] = self._grab_effort(o_d)

        if info.get('success', False):
            print("SUCCESS!")
            action['grab_effort'] = -1
            if not hasattr(self, 'success_halt_counts'):
                self.success_halt_counts = 0
            if self.success_halt_counts < 20:
                print('Halting after success.')
                action['delta_pos'] *= 0
                self.success_halt_counts += 1

        return action.array

    @staticmethod
    def _desired_pos(o_d, info=None):
        if info is None:
            info = {}
        pos_curr = o_d["hand_pos"]
        pos_block = o_d["block_pos"] + np.array([-0.005, 0.0, 0.015])
        pos_block_ = o_d["block_pos"]
        pos_shelf_x = o_d["shelf_x"]
        pos_shelf_y = o_d["shelf_yz"][0]
        pos_shelf_z = 0.265
        pos_shelf = np.array([o_d["shelf_x"], o_d["shelf_yz"][0], 0.265])

        print(pos_block)
        print(pos_curr)
        print(pos_shelf)
        
        success = info.get('success', False)
        if success:
            state = 'Leaving cup.'
            print(state)
            return np.array([pos_curr[0], pos_curr[1], pos_curr[2] + 0.3])
        elif np.linalg.norm(pos_curr[:2] - pos_block[:2]) > 0.04:
            # positioning over block
            print(1)
            return pos_block + np.array([0.0, 0.0, 0.265])
        elif abs(pos_curr[2] - pos_block[2]) > 0.05:
            # grabbing block
            print(2)
            return pos_block
        elif np.abs(pos_block[2] - pos_shelf_z) > 0.02:
            # centering with goal pos
            print(3)
            return np.array([pos_curr[0], pos_curr[1], pos_shelf[2] - pos_block[2] + pos_curr[2]])
        elif np.abs(pos_curr[0] - pos_shelf_x) > 0.01:
            # centering with goal pos
            print(4)
            return np.array([pos_shelf_x, pos_curr[1], 0.265])
        elif np.abs(pos_curr[1] - pos_shelf_y) > 0.01:
            # centering with goal pos
            print(5)
            return np.array([pos_curr[0], pos_shelf_y, pos_shelf[2] - pos_block[2] + pos_curr[2]])
        elif pos_curr[2] < 0.265:
            print(6)
            # move up to correct height
            pos_new = pos_curr + np.array([0.0, 0.0, 0.265])
            return pos_new
        else:
            # move forward to goal
            print(7)
            return pos_shelf - pos_block_ + pos_curr
            # pos_new = pos_curr + np.array([0., 0.05, 0.])
            # return pos_new

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d["hand_pos"]
        pos_block = o_d["block_pos"]

        if (
            np.linalg.norm(pos_curr[:2] - pos_block[:2]) > 0.04
            or abs(pos_curr[2] - pos_block[2]) > 0.15
        ):
            return -1.0

        else:
            return 0.7
