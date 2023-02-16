from task_config import TASK_DICK
import numpy as np
from PIL import Image
import os
from pathlib import Path

CAMERA_LIST = ["corner3", "corner", "corner2", "topview", "gripperPOV", "behindGripper"]

def show_all_task():
    return TASK_DICK.keys()

def save_jpeg(image_array, save_path):
    im = Image.fromarray(image_array)
    im.save(save_path)

def get_img(env):
    image = {}
    for camera_name in CAMERA_LIST:
        img = env.render(offscreen=True, camera_name=camera_name)
        image[camera_name] = img
    return image

def show_demo(task_name, seed, demo):
    img_dict = demo['img']
    for camera_name in CAMERA_LIST:
        img_list = img_dict[camera_name]
        root_path = Path(__file__).parent / 'data' / task_name / str(seed) / camera_name
        root_path.mkdir(exist_ok=True, parents=True)
        # print(len(img_list))
        for i, img in enumerate(img_list):
            save_jpeg(img, root_path / (str(i)+".jpeg"))

def run_demo(task_name, seed=0):
    env = TASK_DICK[task_name]['env'](seed=seed)
    policy = TASK_DICK[task_name]['policy']()

    demo = {
        'obs': [],
        'action': [],
        'reward': [],
        'done': [],
        'img': {}
    }

    obs = env.reset()
    demo['obs'].append(obs)

    img_dict = get_img(env)
    for k, v in img_dict.items():
        demo['img'][k] = [v]

    done = False
    step = 0

    while not done and step < 499:
        a = policy.get_action(obs)
        a = np.clip(a, -1, 1)
        demo['action'].append(a)

        obs, reward, done, info = env.step(a)
        done = info['success']
        demo['obs'].append(obs)
        demo['reward'].append(reward)
        demo['done'].append(done)

        img_dict = get_img(env)
        for k, v in img_dict.items():
            demo['img'][k].append(v)

        step += 1
    
    print(task_name, seed, "done", done)
    
    return demo

if __name__ == "__main__":
    for task_name in TASK_DICK.keys():
        demo = run_demo(task_name=task_name, seed=0)
        show_demo(task_name=task_name, seed=0, demo=demo)
