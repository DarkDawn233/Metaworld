
import numpy as np
import random
from PIL import Image
import imageio
import os
from pathlib import Path
import h5py
import threading
import multiprocessing
import pickle
from typing import NamedTuple
from metaworld.envs.mujoco.sawyer_xyz.display.sawyer_mug_display import SawyerMugEnvV2Display

CAMERA_LIST = ["corner3", "corner", "corner2", "topview", "behindGripper"]
# CAMERA_LIST = ["corner3", "corner", "topview"]

class Task(NamedTuple):
    """All data necessary to describe a single MDP.

    Should be passed into a MetaWorldEnv's set_task method.
    """

    env_name: str
    data: bytes  # Contains env parameters like random_init and *a* goal
    
def encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))

def save_jpeg(image_array, save_path):
    im = Image.fromarray(image_array)
    im.save(save_path)

def get_img(env):
    image = {}
    for camera_name in CAMERA_LIST:
        img = env.render(offscreen=True, camera_name=camera_name)
        image[camera_name] = img
    return image

def show_demo(demo, pos):
    img_dict = demo['img']
    pos_str = ','.join(["{:.2f}".format(pos_i) for pos_i in pos])
    for camera_name in CAMERA_LIST:
        img = img_dict[camera_name]
        root_path = Path(__file__).parent / 'data' / 'pos_data' / pos_str 
        root_path.mkdir(exist_ok=True, parents=True)
        save_jpeg(img, root_path / (camera_name+".jpeg"))

def run_demo(pos=None):
    env = SawyerMugEnvV2Display(pos)
    env.random_init = False
    env._freeze_rand_vec = True
    env._set_task_called = True
    kwargs = {}
    env._set_task_inner(**kwargs)
    rand_vec = env._last_rand_vec
    kwargs.update(dict(rand_vec=rand_vec, env_cls=SawyerMugEnvV2Display, partially_observable=False))
    task = encode_task('test', kwargs)
    # env = SawyerMugEnvV2Display(pos)
    env.set_task(task)
    demo = {
        'obs': None,
        'img': {}
    }

    obs = env.reset()
    demo['obs'] = obs

    img_dict = get_img(env)
    for k, v in img_dict.items():
        demo['img'][k] = v

    return demo


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    task_name = "mug-display"
    
    posx_list = [-0.6, 0.0, 0.6]
    posy_list = [0.3, 0.6, 0.9]
    
    for x in posx_list:
        for y in posy_list:
            pos = [x, y, 0]
            demo = run_demo(pos=pos)
            show_demo(demo, pos)
        # show_demo(task_name=task_name, seed=seed, demo=demo, gif=True)
