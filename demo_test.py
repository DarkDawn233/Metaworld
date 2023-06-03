from task_config import TASK_DICK
import numpy as np
import random
import imageio
from pathlib import Path
from typing import List, Dict
import copy
import logging
import time
from metaworld.envs.env_utils import get_logger


logger = get_logger(__name__)


CAMERA_LIST = ["corner3", "corner", "corner2", "topview"]

class Demo(object):
    def __init__(self, task_name, seed=None, fix_reset=True, save_gif=False) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.env = TASK_DICK[task_name]['env'](seed=seed)
        self.policy = TASK_DICK[task_name]['policy']()
        self.obs = self.env.reset()
        if fix_reset:
            self.env.fix_reset()
        self.info = {}
        self.done = False
        self.obs_img = None
        self.step = 0
        self.task_step = 0
        self.task_max_step = 500
        self.now_task = None

        self.save_gif = save_gif
        if self.save_gif:
            self.img_list = []
        
        self.task_num = 0
        self.gif_file_name = time.strftime("%Y-%m-%d,%H:%M:%S",time.gmtime())
    
    def reset_task_list(self, task_list : List[str]) -> None:
        """
        该接口暂时不使用
        重置任务列表/终止任务
        """
        assert isinstance(task_list, list)
        logger.info(f"reset_task_list: {task_list}")
        self.env.reset_task_list(task_list)
    
    def append_task_list(self, task_list : List[str]) -> None:
        """
        在当前任务列表后添加任务列表
        """
        assert isinstance(task_list, list)
        logger.info(f"append_task_list: {task_list}")
        self.env.append_task_list(task_list)
    
    def reset(self) -> None:
        """
        重置环境
        """
        # if self.now_task is not None:
        self.gif_save(str(self.task_num) + '-' + str(self.now_task) + '-' + 'last')
        self.task_num += 1

        self.obs = self.env.reset()
        self.info = {}
        self.done = False
        self.obs_img = None
        self.step = 0
        self.task_step = 0
        self.now_task = None
        # 重置策略 / 考虑模型是否有重置操作
        self.policy.reset()
        
        if self.save_gif:
            self.img_list = []

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
        # logger.info(self.env.task_list)
        self.obs_img = self._get_obs_img()
        # TODO 模型前向代替policy
        action = self.policy.get_action(self.obs, self.now_task, self.info)
        action = np.clip(action, -1, 1)

        last_task = self.info.get('task_name', None)
        self.obs, reward, self.done, self.info = self.env.step(action)
        self.now_task = self.info['task_name']
        self.step += 1
        if last_task != self.now_task:
            self.task_step = 0
            self.gif_save(str(self.task_num) + '-' + str(last_task))
            self.task_num += 1

        else:
            self.task_step += 1
        return self._get_demo_img()

    def over(self) -> bool:
        """
        判断任务失败
        """
        if self.info.get('task_name', None) is None:
            return False
        over_f = self.task_step > self.task_max_step
        if over_f:
            self.gif_save(str(self.task_num) + '-' + self.now_task + '-' + 'fail')
        return over_f
    
    def gif_save(self, name : str = None) -> None:
        """
        当前 img_list 存成 gif 文件
        """
        if name is None:
            name = 'demo'
        if self.save_gif:
            root_path = Path(__file__).parent / 'data_demo' / self.gif_file_name
            root_path.mkdir(exist_ok=True, parents=True)
            file_path = root_path / (name + '.gif')
            logger.info(f'saving gif {file_path}')
            imageio.mimsave(str(file_path), self.img_list, duration=40)
            self.img_list = []

    def read_states(self) -> str:
        return self.env.read_states()


if __name__ == "__main__":
    task_name = 'display-3d3m'
    seed = 0
    demo = Demo(task_name=task_name, seed=seed, fix_reset=True, save_gif=True)
    # test_task_dict = {
    #     10: ['(green)drawer-open', '(gray)drawer-open', '(blue)desk-pick', 
    #          '()coffee-push', '()coffee-button', '()coffee-pull', '(green)drawer-place'],
    #     1000: 'stop',
    # }
    test_task_dict = {
        10: ['(green)drawer-open', '(blue)desk-pick', '(green)drawer-place',
             '(white)desk-pick', '(green)drawer-place',],
        2000: 'stop',
    }
    
    step = 0
    while True:
        """
        目前暂不支持对书架进行操作
        支持的任务列表：[
            'coffee-button' 按咖啡机开关,
            'coffee-pull'   从咖啡机取走咖啡杯,
            'coffee-push'   放置咖啡杯至咖啡机,
            'drawer-close'  关闭抽屉,
            'drawer-open'   打开抽屉,
            'drawer-pick'   从抽屉取走咖啡杯,
            'drawer-place'  放置咖啡杯到抽屉,
            'desk-pick'     从桌面取走咖啡杯,
            'desk-place'    放置咖啡杯到桌上,
            'bin-pick'      从盒子里取走咖啡杯,
            'bin-place'     放置咖啡杯到盒子里,
            'reset'         机械臂复位
            ]
        支持的指令：
        'reset':    重置环境
        'stop':     停止环境运行
        """
        img = demo.env_step()
        # TODO 推流
        # TODO 获取指令instruction
        # reset 表示重置环境 stop 表示停止演示
        instruction = test_task_dict.get(step, None)
        if isinstance(instruction, list):
            demo.append_task_list(instruction)
        elif isinstance(instruction, str):
            if instruction == 'reset':
                demo.reset()
            elif instruction == 'stop':
                demo.gif_save('stop')
                break
            else:
                raise ValueError(f"Error instruction: {instruction}")
        step += 1
        if demo.over():
            raise Exception(f"Task {demo.now_task} Policy failed.")
    # demo.gif_save(name=time.strftime("%H:%M:%S",time.gmtime()))
    