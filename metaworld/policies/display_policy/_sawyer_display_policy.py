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

from metaworld.policies.display_policy.sawyer_place_display_policy import SawyerPlaceV2DisplayPolicy
from metaworld.policies.display_policy.sawyer_pick_display_policy import SawyerPickV2DisplayPolicy

from metaworld.envs.display_utils import TASKS


class SawyerV2DisplayPolicy(Policy):

    def __init__(self):
        self.policy = None
        self.task_name = None
        self.last_action = None

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
            if now_task == TASKS.COFFEE_BUTTON:
                self.policy = SawyerCoffeeButtonV2DisplayPolicy()
            elif now_task == TASKS.COFFEE_PULL:
                # self.policy = SawyerCoffeePullV2DisplayPolicy()
                self.policy = SawyerPickV2DisplayPolicy()
            elif now_task == TASKS.COFFEE_PUSH:
                # self.policy = SawyerCoffeePushV2DisplayPolicy()
                self.policy = SawyerPlaceV2DisplayPolicy()
            elif now_task == TASKS.DRAWER_CLOSE:
                self.policy = SawyerDrawerCloseV2DisplayPolicy()
            elif now_task == TASKS.DRAWER_OPEN:
                self.policy = SawyerDrawerOpenV2DisplayPolicy()
            elif now_task == TASKS.DRAWER_PICK:
                # self.policy = SawyerDrawerPickV2DisplayPolicy()
                self.policy = SawyerPickV2DisplayPolicy()
            elif now_task == TASKS.DRAWER_PLACE:
                # self.policy = SawyerDrawerPlaceV2DisplayPolicy()
                self.policy = SawyerPlaceV2DisplayPolicy()
            elif now_task == TASKS.DESK_PICK:
                # self.policy = SawyerDeskPickV2DisplayPolicy()
                self.policy = SawyerPickV2DisplayPolicy()
            elif now_task == TASKS.DESK_PLACE:
                # self.policy = SawyerDeskPlaceV2DisplayPolicy()
                self.policy = SawyerPlaceV2DisplayPolicy()
            elif now_task == TASKS.BIN_PICK:
                self.policy = SawyerPickV2DisplayPolicy()
            elif now_task == TASKS.BIN_PLACE:
                self.policy = SawyerPlaceV2DisplayPolicy()
            elif now_task == TASKS.RESET:
                self.policy = SawyerResetV2DisplayPolicy()
            else:
                print('Not policy set.')
                self.policy = None
        if self.policy is None:
            if self.last_action is None:
                return np.zeros(4)
            else:
                return np.array([0., 0., 0., 1.]) * self.last_action
        else:
            self.last_action = self.policy.get_action(obs, info)
            return self.last_action
    
    def reset(self):
        self.policy = None
        self.task_name = None
        self.last_action = None