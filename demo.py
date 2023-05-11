from metaworld.task_config import TASK_DICK
import numpy as np
import random
import imageio
from pathlib import Path
from typing import List, Dict
import copy


CAMERA_LIST = ["corner3", "corner", "corner2", "topview"]

class Demo(object):
    def __init__(self, task_name, seed=None, save_gif=False) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.env = TASK_DICK[task_name]['env'](seed=seed)
        self.policy = TASK_DICK[task_name]['policy']()
        self.obs = self.env.reset()
        self.info = {}
        self.done = False
        self.obs_img = None
        self.step = 0
        self.max_step = 1000
        self.now_task = None

        self.save_gif = save_gif
        if self.save_gif:
            self.img_list = []
    
    def reset_task_list(self, task_list : List[str]) -> None:
        """
        重置任务列表/终止任务
        """
        print(f"reset_task_list: {task_list}")
        self.env.reset_task_list(task_list)

    def _get_obs_img(self) -> Dict[str, np.ndarray]:
        img_dict = {}
        for camera_name in ["corner3", "corner", "topview"]:
            img = self.env.render(offscreen=True, camera_name=camera_name, resolution=(320, 240))
            img_dict[camera_name] = img
        return img_dict
    
    def _get_demo_img(self) -> np.ndarray:
        img = self.env.render(offscreen=True, camera_name='corner3')
        if self.save_gif:
            self.img_list.append(img)
        return img

    def env_step(self) -> np.ndarray:
        print(self.env.task_list)
        self.obs_img = self._get_obs_img()
        action = self.policy.get_action(self.obs, self.now_task, self.info)
        action = np.clip(action, -1, 1)
        self.obs, reward, self.done, self.info = self.env.step(action)
        self.now_task = self.info['task_name']
        self.step += 1
        return self._get_demo_img()

    def over(self) -> bool:
        done = (len(self.env.task_list) == 0) or (self.step > self.max_step)
        return done
    
    def gif_save(self) -> None:
        if self.save_gif:
            print(f'saving gif at {self.step} ...')
            root_path = Path(__file__).parent / 'data'
            root_path.mkdir(exist_ok=True, parents=True)
            imageio.mimsave(str(root_path / ('demo.gif')), self.img_list, duration=0.04)
    

if __name__ == "__main__":
    task_name = 'display'
    seed = 0
    demo = Demo(task_name=task_name, seed=seed, save_gif=True)
    # task_list = []
    task_list1 = ['drawer-open', 'drawer-place', 'drawer-close']
    wait_step = 30
    task_list2 = ['drawer-open', 'drawer-pick']
    done = False
    while not done:
        img = demo.env_step()
        # TODO 推流
        if demo.over():
            # 设置task_list
            if task_list1 is not None:
                demo.reset_task_list(task_list=copy.deepcopy(task_list1))
                task_list1 = None
            elif task_list2 is not None:
                if wait_step > 0:
                    wait_step -= 1
                    continue
                demo.reset_task_list(task_list=copy.deepcopy(task_list2))
                task_list2 = None
            else:
                print(f'demo over step_num {demo.step}')
                demo.gif_save()
                done = True
    