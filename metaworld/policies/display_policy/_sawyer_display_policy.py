import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.envs.display_utils import QUAT_LIST

from metaworld.policies.display_policy.sawyer_coffee_button_display_policy import SawyerCoffeeButtonV2DisplayPolicy
from metaworld.policies.display_policy.sawyer_coffee_pull_display_policy import SawyerCoffeePullV2DisplayPolicy
from metaworld.policies.display_policy.sawyer_coffee_push_display_policy import SawyerCoffeePushV2DisplayPolicy
from metaworld.policies.display_policy.sawyer_drawer_close_display_policy import SawyerDrawerCloseV2DisplayPolicy
from metaworld.policies.display_policy.sawyer_drawer_open_display_policy import SawyerDrawerOpenV2DisplayPolicy
from metaworld.policies.display_policy.sawyer_shelf_place_display_policy import SawyerShelfPlaceV2DisplayPolicy
from metaworld.policies.display_policy.sawyer_desk_pick_display_policy import SawyerDeskPickV2DisplayPolicy
from metaworld.policies.display_policy.sawyer_desk_place_display_policy import SawyerDeskPlaceV2DisplayPolicy
from metaworld.policies.display_policy.sawyer_reset_display_policy import SawyerResetV2DisplayPolicy

from metaworld.policies.display_policy.sawyer_drawer_place_display_policy import SawyerDrawerPlaceV2DisplayPolicy
from metaworld.policies.display_policy.sawyer_drawer_pick_display_policy import SawyerDrawerPickV2DisplayPolicy

class SawyerV2DisplayPolicy(Policy):

    def __init__(self):
        self.policy = None
        self.task_name = None

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

    def get_action(self, obs, now_task, info={}):
        if now_task != self.task_name:
            self.task_name = now_task
            if now_task == 'coffee-button':
                self.policy = SawyerCoffeeButtonV2DisplayPolicy()
            elif now_task == 'coffee-pull':
                self.policy = SawyerCoffeePullV2DisplayPolicy()
            elif now_task == 'coffee-push':
                self.policy = SawyerCoffeePushV2DisplayPolicy()
            elif now_task == 'drawer-close':
                self.policy = SawyerDrawerCloseV2DisplayPolicy()
            elif now_task == 'drawer-open':
                self.policy = SawyerDrawerOpenV2DisplayPolicy()
            elif now_task == 'drawer-pick':
                self.policy = SawyerDrawerPickV2DisplayPolicy()
            elif now_task == 'drawer-place':
                self.policy = SawyerDrawerPlaceV2DisplayPolicy()
            elif now_task == 'desk-pick':
                self.policy = SawyerDeskPickV2DisplayPolicy()
            elif now_task == 'desk-place':
                self.policy = SawyerDeskPlaceV2DisplayPolicy()
            elif now_task == 'reset':
                self.policy = SawyerResetV2DisplayPolicy()
            else:
                print('Not policy set.')
                self.policy = None
        if self.policy is None:
            return np.zeros(4)
        else:
            return self.policy.get_action(obs, info)