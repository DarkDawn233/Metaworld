from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import *
from metaworld.policies.display_policy import *

from metaworld.envs import ALL_DISPLAY_ENVIRONMENTS_GOAL_OBSERVABLE

TASK_DICK = {
     'assembly':                    {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["assembly-v2-goal-observable"]                      ,'policy': SawyerAssemblyV2Policy},
     'basketball':                  {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["basketball-v2-goal-observable"]                    ,'policy': SawyerBasketballV2Policy},
     'bin-picking':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["bin-picking-v2-goal-observable"]                   ,'policy': SawyerBinPickingV2Policy},
     'box-close':                   {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["box-close-v2-goal-observable"]                     ,'policy': SawyerBoxCloseV2Policy},
     'button-press-topdown':        {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-topdown-v2-goal-observable"]          ,'policy': SawyerButtonPressTopdownV2Policy},
     'button-press-topdown-wall':   {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-topdown-wall-v2-goal-observable"]     ,'policy': SawyerButtonPressTopdownWallV2Policy},
     'button-press':                {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-v2-goal-observable"]                  ,'policy': SawyerButtonPressV2Policy},
     'button-press-wall':           {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-wall-v2-goal-observable"]             ,'policy': SawyerButtonPressWallV2Policy},
     'coffee-button':               {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["coffee-button-v2-goal-observable"]                 ,'policy': SawyerCoffeeButtonV2Policy},
     'coffee-pull':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["coffee-pull-v2-goal-observable"]                   ,'policy': SawyerCoffeePullV2Policy},
     'coffee-push':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["coffee-push-v2-goal-observable"]                   ,'policy': SawyerCoffeePushV2Policy},
     'dial-turn':                   {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["dial-turn-v2-goal-observable"]                     ,'policy': SawyerDialTurnV2Policy},
     'disassemble':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["disassemble-v2-goal-observable"]                   ,'policy': SawyerDisassembleV2Policy},
     'door-close':                  {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-close-v2-goal-observable"]                    ,'policy': SawyerDoorCloseV2Policy},
     'door-lock':                   {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-lock-v2-goal-observable"]                     ,'policy': SawyerDoorLockV2Policy},
     'door-open':                   {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]                     ,'policy': SawyerDoorOpenV2Policy},
     'door-unlock':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-unlock-v2-goal-observable"]                   ,'policy': SawyerDoorUnlockV2Policy},
     'drawer-close':                {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-close-v2-goal-observable"]                  ,'policy': SawyerDrawerCloseV2Policy},
     'drawer-open':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-open-v2-goal-observable"]                   ,'policy': SawyerDrawerOpenV2Policy},
     'faucet-close':                {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["faucet-close-v2-goal-observable"]                  ,'policy': SawyerFaucetCloseV2Policy},
     'faucet-open':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["faucet-open-v2-goal-observable"]                   ,'policy': SawyerFaucetOpenV2Policy},
     'hammer':                      {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["hammer-v2-goal-observable"]                        ,'policy': SawyerHammerV2Policy},
     'hand-insert':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["hand-insert-v2-goal-observable"]                   ,'policy': SawyerHandInsertV2Policy},
     'handle-press-side':           {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["handle-press-side-v2-goal-observable"]             ,'policy': SawyerHandlePressSideV2Policy},
     'handle-press':                {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["handle-press-v2-goal-observable"]                  ,'policy': SawyerHandlePressV2Policy},
     'handle-pull-side':            {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["handle-pull-side-v2-goal-observable"]              ,'policy': SawyerHandlePullSideV2Policy},
     'handle-pull':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["handle-pull-v2-goal-observable"]                   ,'policy': SawyerHandlePullV2Policy},
     'lever-pull':                  {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["lever-pull-v2-goal-observable"]                    ,'policy': SawyerLeverPullV2Policy},
     'peg-insert-side':             {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["peg-insert-side-v2-goal-observable"]               ,'policy': SawyerPegInsertionSideV2Policy},
     'peg-unplug-side':             {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["peg-unplug-side-v2-goal-observable"]               ,'policy': SawyerPegUnplugSideV2Policy},
     'pick-out-of-hole':            {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["pick-out-of-hole-v2-goal-observable"]              ,'policy': SawyerPickOutOfHoleV2Policy},
     'pick-place':                  {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["pick-place-v2-goal-observable"]                    ,'policy': SawyerPickPlaceV2Policy},
     'pick-place-wall':             {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["pick-place-wall-v2-goal-observable"]               ,'policy': SawyerPickPlaceWallV2Policy},
     'plate-slide-back-side':       {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["plate-slide-back-side-v2-goal-observable"]         ,'policy': SawyerPlateSlideBackSideV2Policy},
     'plate-slide-back':            {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["plate-slide-back-v2-goal-observable"]              ,'policy': SawyerPlateSlideBackV2Policy},
     'plate-slide-side':            {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["plate-slide-side-v2-goal-observable"]              ,'policy': SawyerPlateSlideSideV2Policy},
     'plate-slide':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["plate-slide-v2-goal-observable"]                   ,'policy': SawyerPlateSlideV2Policy},
     'push-back':                   {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["push-back-v2-goal-observable"]                     ,'policy': SawyerPushBackV2Policy},
     'push':                        {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["push-v2-goal-observable"]                          ,'policy': SawyerPushV2Policy},
     'push-wall':                   {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["push-wall-v2-goal-observable"]                     ,'policy': SawyerPushWallV2Policy},
     'reach':                       {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["reach-v2-goal-observable"]                         ,'policy': SawyerReachV2Policy},
     'reach-wall':                  {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["reach-wall-v2-goal-observable"]                    ,'policy': SawyerReachWallV2Policy},
     'shelf-place':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["shelf-place-v2-goal-observable"]                   ,'policy': SawyerShelfPlaceV2Policy},
     'soccer':                      {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["soccer-v2-goal-observable"]                        ,'policy': SawyerSoccerV2Policy},
     'stick-pull':                  {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["stick-pull-v2-goal-observable"]                    ,'policy': SawyerStickPullV2Policy},
     'stick-push':                  {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["stick-push-v2-goal-observable"]                    ,'policy': SawyerStickPushV2Policy},
     'sweep-into':                  {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["sweep-into-v2-goal-observable"]                    ,'policy': SawyerSweepIntoV2Policy},
     'sweep':                       {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["sweep-v2-goal-observable"]                         ,'policy': SawyerSweepV2Policy},
     'window-close':                {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["window-close-v2-goal-observable"]                  ,'policy': SawyerWindowCloseV2Policy},
     'window-open':                 {'env': ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["window-open-v2-goal-observable"]                   ,'policy': SawyerWindowOpenV2Policy},

     'coffee-button-display':       {'env': ALL_DISPLAY_ENVIRONMENTS_GOAL_OBSERVABLE["coffee-button-v2-display-goal-observable"]    ,'policy': SawyerCoffeeButtonV2DisplayPolicy},
     'coffee-pull-display':         {'env': ALL_DISPLAY_ENVIRONMENTS_GOAL_OBSERVABLE["coffee-pull-v2-display-goal-observable"]      ,'policy': SawyerCoffeePullV2DisplayPolicy},
     'coffee-push-display':         {'env': ALL_DISPLAY_ENVIRONMENTS_GOAL_OBSERVABLE["coffee-push-v2-display-goal-observable"]      ,'policy': SawyerCoffeePushV2DisplayPolicy},
     'drawer-close-display':        {'env': ALL_DISPLAY_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-close-v2-display-goal-observable"]     ,'policy': SawyerDrawerCloseV2DisplayPolicy},
     'drawer-open-display':         {'env': ALL_DISPLAY_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-open-v2-display-goal-observable"]      ,'policy': SawyerDrawerOpenV2DisplayPolicy},
     'shelf-place-display':         {'env': ALL_DISPLAY_ENVIRONMENTS_GOAL_OBSERVABLE["shelf-place-v2-display-goal-observable"]      ,'policy': SawyerShelfPlaceV2DisplayPolicy},

     'drawer-place-display':        {'env': ALL_DISPLAY_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-place-v2-display-goal-observable"]     ,'policy': SawyerDrawerPlaceV2DisplayPolicy},
     'drawer-pick-display':         {'env': ALL_DISPLAY_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-pick-v2-display-goal-observable"]      ,'policy': SawyerDrawerPickV2DisplayPolicy},

     'display':                     {'env': ALL_DISPLAY_ENVIRONMENTS_GOAL_OBSERVABLE["v2-display-goal-observable"]                  ,'policy': SawyerV2DisplayPolicy},
     
}

def test():
    for task_name, task in TASK_DICK.items():
        env = task['env'](seed=0)
        policy = task['policy']()
        step = 0
        obs = env.reset()
        done = False
        while step < 499 and not done:
            a = policy.get_action(obs)
            obs, reward, done, info = env.step(a)
            done = info['success']
            step += 1
        print(task_name, ":", done)

if __name__ == "__main__":
    test()