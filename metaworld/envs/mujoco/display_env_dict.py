from collections import OrderedDict
import re
import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.display import (
    SawyerEnvV2Display,
    SawyerCoffeeButtonEnvV2Display,
    SawyerCoffeePullEnvV2Display,
    SawyerCoffeePushEnvV2Display,
    SawyerDrawerCloseEnvV2Display,
    SawyerDrawerOpenEnvV2Display,
    SawyerShelfPlaceEnvV2Display,
    SawyerResetEnvV2Display,
    SawyerDeskPickEnvV2Display,
    SawyerDeskPlaceEnvV2Display,

    SawyerDrawerPlaceEnvV2Display,
    SawyerDrawerPickEnvV2Display
)

ALL_DISPLAY_ENVIRONMENTS = OrderedDict((
    ('v2-display', SawyerEnvV2Display),

    ('coffee-button-v2-display', SawyerCoffeeButtonEnvV2Display),
    ('coffee-pull-v2-display', SawyerCoffeePullEnvV2Display),
    ('coffee-push-v2-display', SawyerCoffeePushEnvV2Display),
    ('drawer-close-v2-display', SawyerDrawerCloseEnvV2Display),
    ('drawer-open-v2-display', SawyerDrawerOpenEnvV2Display),
    ('shelf-place-v2-display', SawyerShelfPlaceEnvV2Display),
    ('reset-v2-display', SawyerResetEnvV2Display),
    ('desk-pick-v2-display', SawyerDeskPickEnvV2Display),
    ('desk-place-v2-display', SawyerDeskPlaceEnvV2Display),

    ('drawer-place-v2-display', SawyerDrawerPlaceEnvV2Display),
    ('drawer-pick-v2-display', SawyerDrawerPickEnvV2Display),
))

def create_observable_goal_envs():
    observable_goal_envs = {}
    for env_name, env_cls in ALL_DISPLAY_ENVIRONMENTS.items():
        d = {}

        def initialize(env, seed=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()
            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed)
                np.random.set_state(st0)

        d['__init__'] = initialize
        og_env_name = re.sub("(^|[-])\s*([a-zA-Z])",
                             lambda p: p.group(0).upper(), env_name)
        og_env_name = og_env_name.replace("-", "")

        og_env_key = '{}-goal-observable'.format(env_name)
        og_env_name = '{}GoalObservable'.format(og_env_name)
        ObservableGoalEnvCls = type(og_env_name, (env_cls, ), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)

ALL_DISPLAY_ENVIRONMENTS_GOAL_OBSERVABLE = create_observable_goal_envs()