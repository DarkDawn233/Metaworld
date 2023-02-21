from task_config import TASK_DICK
import numpy as np
from PIL import Image
import os
from pathlib import Path
import h5py

# CAMERA_LIST = ["corner3", "corner", "corner2", "topview", "behindGripper"]
CAMERA_LIST = ["corner3", "corner", "topview"]

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
    ever_done = False
    final_done = 0
    step = 0

    while final_done < 1 and step < 300:
        a = policy.get_action(obs)
        a = np.clip(a, -1, 1)
        demo['action'].append(a)
        obs, reward, done, info = env.step(a)
        if done:
            ever_done = True
        if ever_done:
            final_done += 1
        # done = info['success']
        demo['obs'].append(obs)
        demo['reward'].append(reward)
        demo['done'].append(done)

        img_dict = get_img(env)
        for k, v in img_dict.items():
            demo['img'][k].append(v)

        step += 1
        print("step:", step, "reward:", reward, env.last_reward, "a:", a)
    
    print(task_name, seed, "done", done, step)
    
    return demo, done

def write_h5(task_name, seed, demo):
    root_path = Path(__file__).parent / 'data' / task_name
    root_path.mkdir(exist_ok=True, parents=True)
    file_path = root_path / (str(seed) + ".hdf5")
    f = h5py.File(file_path, "w")
    for k, v in demo.items():
        if isinstance(v, dict):
            g = f.create_group(k)
            for vk, vv in v.items():
                g.create_dataset(vk, data=vv, compression='gzip')
        else:
            f.create_dataset(k, data=v,compression='gzip')
    f.close()
    
def read_h5(task_name, seed):
    file_path = Path(__file__).parent / 'data' / task_name / (str(seed) + ".hdf5")
    f = h5py.File(file_path, "r")
    data = {}
    for k, v in f.items():
        # print(k, type(v))
        if isinstance(v, h5py._hl.group.Group):
            data[v] = {}
            for vk, vv in v.items():
                data[v][vk] = np.array(vv)
                # print(vk, data[v][vk].shape)
        else:
            data[k] = np.array(v)
            # print(k, data[k].shape)
    return data

def test_env(task_name):
    fail_list = []
    for seed in range(100):
        demo, success = run_demo(task_name=task_name, seed=seed)
        if not success:
            fail_list.append(seed)
    print("Fail list:", fail_list)

if __name__ == "__main__":
    # Fail list: [12, 29, 45, 96]
    task_name = "coffee-pull"
    seed = 0
    demo, _ = run_demo(task_name=task_name, seed=seed)

    show_demo(task_name=task_name, seed=seed, demo=demo)

    # write_h5(task_name=task_name, seed=seed, demo=demo)
    # read_h5(task_name=task_name, seed=seed)

    # test_env(task_name)
