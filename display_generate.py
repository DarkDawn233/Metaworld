from task_config import TASK_DICK
import numpy as np
import random
from PIL import Image
import imageio
import os
import copy
from pathlib import Path
import h5py
import threading
import multiprocessing
# import cv2

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
        img = env.render(offscreen=True, camera_name=camera_name, resolution=(320, 240))
        image[camera_name] = img
    return image

def show_demo(task_name, seed, demo, gif=False):
    img_dict = demo['img']
    for camera_name in CAMERA_LIST:
        img_list = img_dict[camera_name]
        root_path = Path(__file__).parent / 'data' / task_name / str(seed)
        root_path.mkdir(exist_ok=True, parents=True)
        if gif:
            imageio.mimsave(str(root_path / (camera_name + '.gif')), img_list, fps=25)
            continue
        root_path = root_path / camera_name
        root_path.mkdir(exist_ok=True, parents=True)
        for i, img in enumerate(img_list):
            save_jpeg(img, root_path / (str(i)+".jpeg"))

def run_demo(task_name, task_list=[], seed=0, max_step=500, debug=False):
    env = TASK_DICK[task_name]['env'](seed=seed)
    policy = TASK_DICK[task_name]['policy']()

    demo = {
        'obs': [],
        'action': [],
        'reward': [],
        'src_reward': [],
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
    info = {}
    now_task = None

    env.reset_task_list(task_list)
    result_done = []

    while len(env.task_list) > 0 and step < max_step:
        a = policy.get_action(obs, now_task, info)
        a = np.clip(a, -1, 1)
        demo['action'].append(a)
        obs, reward, done, info = env.step(a)
        now_task = info['task_name']
        if done:
            result_done.append(now_task)
        # done = info['success']
        demo['obs'].append(obs)
        demo['reward'].append(reward)
        demo['src_reward'].append(env.last_reward)
        demo['done'].append(done)

        img_dict = get_img(env)
        for k, v in img_dict.items():
            demo['img'][k].append(v)

        step += 1
        if debug:
            print("step:", step, "reward:", reward, env.last_reward, "a:", a)
    
    print(task_name, seed, "done", result_done, step)
    
    return demo, done

def cal_return_to_go(demo):
    reward = demo['reward']
    done = demo['done']
    returns = []
    ret = 0.
    for r, d in zip(reward, done):
        ret = ret * (1 - d) + r
        returns.append(ret)
    returns.reverse()
    demo['return'] = returns
    return demo

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

def exist_h5(task_name, seed):
    root_path = Path(__file__).parent / 'data' / task_name
    file_path = root_path / (str(seed) + ".hdf5")
    if file_path.exists():
        return True
    else:
        return False
    
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
                # print(vk, data[v][vk].shape, data[v][vk].dtype)
        else:
            data[k] = np.array(v)
            # print(k, data[k].shape, data[k].dtype)
            # print(np.array(v[1]))
    return data

def thread_test_env(task_name, begin_seed, end_seed, q):
    for seed in range(begin_seed, end_seed):
        demo, success = run_demo(task_name=task_name, seed=seed)
        if not success:
            q.put(seed)

def test_env(task_name, thread_num=20):
    thread_len = 100 // thread_num
    q = multiprocessing.Queue()
    thread_list = []
    for t_id in range(thread_num):
        t = multiprocessing.Process(target=thread_test_env, args=(task_name, t_id*thread_len, (t_id+1)*thread_len, q))
        t.start()
        thread_list.append(t)
    for t in thread_list:
        t.join()

    fail_list = []
    while not q.empty():
        fail_seed = q.get()
        fail_list.append(fail_seed)
    fail_list.sort()
    print("Fail list:", fail_list)

def thread_generate_data(t_id, task_name, begin_seed, end_seed):
    root_path = Path(__file__).parent / 'data' / task_name
    root_path.mkdir(exist_ok=True, parents=True)
    success_file_path = root_path / ("fail_" + str(t_id) + ".txt")
    for seed in range(begin_seed, end_seed):
        if exist_h5(task_name=task_name, seed=seed):
            print(f"{task_name} {seed} exist continue.")
            continue
        demo, success = run_demo(task_name=task_name, seed=seed)
        if not success:
            with open(success_file_path, "a") as f:
                f.write(str(seed) + "\n")
        else:
            demo = cal_return_to_go(demo)
            write_h5(task_name=task_name, seed=seed, demo=demo)

def generate_data(task_name, thread_num=10, total_ep=2000):
    thread_list = []
    each_thread_ep = total_ep // thread_num
    for t_id in range(thread_num):
        begin_seed = t_id * each_thread_ep
        end_seed = (t_id + 1) * each_thread_ep
        # t = threading.Thread(target=thread_generate_data, args=(t_id, task_name, begin_seed, end_seed))
        t = multiprocessing.Process(target=thread_generate_data, args=(t_id, task_name, begin_seed, end_seed))
        t.start()
        thread_list.append(t)
    for t in thread_list:
        t.join()

def stat_success(task_name, thread_num=10, total_ep=2000):
    root_path = Path(__file__).parent / 'data' / task_name
    cnt = 0
    for t_id in range(thread_num):
        success_file_path = root_path / ("fail_" + str(t_id) + ".txt")
        if success_file_path.exists():
            with open(success_file_path, "r") as f:
                cnt += len(f.readlines())
    rate = (total_ep-cnt)/total_ep
    print("success_rate:", rate) 
    success_rate_path = root_path / "success_rate.txt"
    with open(success_rate_path, "w") as f:
        f.write("success rate: " + str(rate))

def generate_data_main():
    task_name_list = [
        # "drawer-close-display",
        # "drawer-open-display",
        "drawer-place-display",
        # "drawer-pick-display",
        ]
    for task_name in task_name_list:
        generate_data(task_name=task_name, thread_num=10, total_ep=21000)
        stat_success(task_name=task_name, thread_num=10, total_ep=21000)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    # generate_data_main()
    task_name = "display"
    task_list = ['drawer-open', 'drawer-place', 'drawer-close', 'drawer-open', 'drawer-pick']
    for seed in range(1):
        # seed = 6
        random.seed(seed)
        np.random.seed(seed)
        demo, _ = run_demo(task_name=task_name, task_list=copy.deepcopy(task_list), seed=seed, max_step=1000, debug=True)
        # demo = cal_return_to_go(demo)
        show_demo(task_name=task_name, seed=seed, demo=demo, gif=True)

        # write_h5(task_name=task_name, seed=seed, demo=demo)
    # read_h5(task_name=task_name, seed=seed)

    # test_env(task_name)
    

    
