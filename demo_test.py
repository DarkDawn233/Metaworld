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
from PIL import Image


logger = get_logger(__name__)


CAMERA_LIST = ["corner3", "corner", "corner2", "topview"]

class Demo(object):
    def __init__(self, task_name, seed=None, fix_reset=True, save_gif=False) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.task_name = task_name
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
    
    def get_color_info(self):
        if self.task_name == "display-3d3m":
            return self.env.get_color_info()
        else:
            return None
    
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
        self.full_now_task = None
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
        full_last_task = self.info.get('full_task_name', last_task)
        self.obs, reward, self.done, self.info = self.env.step(action)
        self.now_task = self.info['task_name']
        self.full_now_task = self.info.get('full_task_name', self.now_task)
        self.step += 1
        if full_last_task != self.full_now_task:
            self.task_step = 0
            logger.info(f"{full_last_task} task finish at {self.step}")
            self.gif_save(str(self.task_num) + '-' + str(full_last_task))
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
            self.gif_save(str(self.task_num) + '-' + self.full_now_task + '-' + 'fail')
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
            if len(self.img_list) == 0:
                logger.info(f'no img to save gif {file_path}')
                return 
            imageio.mimsave(str(file_path), self.img_list, duration=40)
            self.img_list = []
    
    def img_save(self, img) -> None:
        root_path = Path(__file__).parent / 'data_demo' / self.gif_file_name
        root_path.mkdir(exist_ok=True, parents=True)
        file_path = root_path / ('now.png')
        im = Image.fromarray(img)
        im.save(file_path)

    def read_states(self) -> str:
        return self.env.read_states()
    
    def empty_task(self) -> bool:
        return len(self.env.task_list) == 0

def rebuild(x):
    color_dict = {
        "bla": "black",  
        "whi": "white",  
        "red": "red",    
        "gre": "green",  
        "blu": "blue",   
        "yel": "yellow", 
        "cya": "cyan",   
        "car": "carmine",
        "gra": "gray",   
        "ora": "orange", 
        "pur": "purple", 
        "bro": "brown",

        "lef": "left",
        "mid": "mid",
        "rig": "right",
    }
    
    task_dict = {
        'cobu': 'coffee-button',
        'coll': 'coffee-pull',
        'cosh': 'coffee-push',
        'drcl': 'drawer-close',
        'drop': 'drawer-open',
        'drpi': 'drawer-pick',
        'drpl': 'drawer-place',
        'depi': 'desk-pick',
        'depl': 'desk-place',
    }
    try:
        if len(x) == 7:
            return '(' + color_dict[x[0:3]] + ')' + task_dict[x[3:3+4]]
        else:
            return '()' + task_dict[x]
    except:
        return None


def interact_test():
    task_name = 'display-3d3m'
    seed = 0
    demo = Demo(task_name=task_name, seed=seed, fix_reset=True, save_gif=True)
    step = 0
    sleep_time = 0
    while True:
        img = demo.env_step()
        if sleep_time > 0:
            sleep_time -= 1
            step += 1
            continue

        if demo.empty_task():
            demo.img_save(img)
            instruction = input("new instruction:")
            if isinstance(instruction, str):
                if "sleep" in instruction:
                    sleep_time = eval(instruction[6:])
                if instruction == 'reset':
                    demo.reset()
                elif instruction == 'stop':
                    demo.gif_save('stop')
                    break
                else:
                    new_instruction = rebuild(instruction)
                    if new_instruction is None:
                        print(f"Error instruction")
                    else:
                        demo.append_task_list([new_instruction])
                    
        step += 1
        if demo.over():
            raise Exception(f"Task {demo.now_task} Policy failed.")

def random_test(seed=0):
    task_name = 'display-3d3m'
    demo = Demo(task_name=task_name, seed=seed, fix_reset=True, save_gif=True)
    demo.env.set_random_generate_task(True)
    step = 0
    sleep_time = 0
    while True:
        img = demo.env_step()           
        step += 1
        if step >= 6000:
            break
        if demo.over():
            # raise Exception(f"Task {demo.full_now_task} Policy failed.")
            print(f"Task {demo.full_now_task} Policy failed.")

def main():
    task_name = 'display-3d3m'
    seed = 0
    demo = Demo(task_name=task_name, seed=seed, fix_reset=True, save_gif=True)
    print(demo.get_color_info())
    # test_task_dict = {
    #     10: ['(green)drawer-open', '(gray)drawer-open', '(blue)desk-pick', 
    #          '()coffee-push', '()coffee-button', '()coffee-pull', '(green)drawer-place'],
    #     1000: 'stop',
    # }
    # test_task_dict = {
    #     10: ['(green)drawer-open', '(blue)desk-pick', '(green)drawer-place',
    #          '(white)desk-pick', '(green)drawer-place',],
    #     2000: 'stop',
    # }
    # test_task_dict = {
    #     10: ['(left)drawer-open'],
    #     200: 'stop'
    # }
    test_task_dict = {
        10: ['(left)drawer-open', '(blue)desk-pick', '(gray)drawer-place',
             '()reset-hand',
             '(green)drawer-open', '(black)desk-pick', 
             '()coffee-push', '()coffee-button', '()coffee-pull', '(mid)drawer-place',
             '(left)drawer-close'],
        3000: 'stop',
    }
    # test_task_dict = {
    #     10: ['()reset-hand'],
    #     200: 'stop'
    # }
    
    step = 0
    sleep_time = 0
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
        if sleep_time > 0:
            sleep_time -= 1
            step += 1
            continue
        # TODO 推流
        # TODO 获取指令instruction
        # reset 表示重置环境 stop 表示停止演示
        instruction = test_task_dict.get(step, None)
        if isinstance(instruction, list):
            demo.append_task_list(instruction)
        elif isinstance(instruction, str):
            if "sleep" in instruction:
                sleep_time = eval(instruction[6:])
            elif instruction == 'reset':
                demo.reset()
            elif instruction == 'stop':
                demo.gif_save('stop')
                break
            else:
                raise ValueError(f"Error instruction: {instruction}")
        step += 1
        if demo.over():
            # print(f"Task {demo.now_task} Policy failed.")
            raise Exception(f"Task {demo.now_task} Policy failed.")
    # demo.gif_save(name=time.strftime("%H:%M:%S",time.gmtime()))

if __name__ == "__main__":
    # interact_test()
    for i in range(0, 20):
        random_test(i)
    