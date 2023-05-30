from task_config import TASK_DICK
import numpy as np
import random
from PIL import Image
import imageio
import os
import copy
from pathlib import Path
import json
import h5py
import threading
import multiprocessing
# import cv2

# CAMERA_LIST = ["corner3", "corner", "corner2", "topview", "behindGripper"]
CAMERA_LIST = ["corner3", "corner", "topview"]
# CAMERA_LIST = ["corner3"]

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
            imageio.mimsave(str(root_path / (camera_name + '.gif')), img_list, duration=40)
            continue
        root_path = root_path / camera_name
        root_path.mkdir(exist_ok=True, parents=True)
        for i, img in enumerate(img_list):
            save_jpeg(img, root_path / (str(i)+".jpeg"))
    if 'task_list' in demo:
        file_path = Path(__file__).parent / 'data' / task_name / str(seed) / 'task.json'
        dict = {
            'task_list' : demo.get('task_list', []),
            'fail_task' : demo.get('fail_task', None)
        }
        with open (file_path, 'w') as f:
            json.dump(dict, f, indent=2)

def clear_demo(demo):
    demo['obs'] = demo['obs'][-1:]
    for k in ['action', 'reward', 'src_reward', 'done']:
        demo[k] = []
    for k, v in demo['img'].items():
        demo['img'][k] = v[-1:]
    return demo

def save_fail_gif(now_task, seed, task_num, demo, fail=True):
    file_name = now_task + '-' + str(seed) + '-' + str(task_num) + '.gif'
    camera_name = 'corner3'
    img_dict = demo['img']
    img_list = img_dict[camera_name]
    if not fail:
        root_path = Path(__file__).parent / 'data' / 'success_gif'
    else:
        root_path = Path(__file__).parent / 'data' / 'fail_gif'
    root_path.mkdir(exist_ok=True, parents=True)
    imageio.mimsave(str(root_path / file_name), img_list, duration=40)


def data_demo(task_list=None, seed=0, max_task_step=500, total_task_num=20, debug=False):
    random.seed(seed)
    np.random.seed(seed)
    env = TASK_DICK['display']['env'](seed=seed)
    policy = TASK_DICK['display']['policy']()

    random_task = task_list is None

    # Env Init
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
    step = 0
    info = {}
    
    task_step = 0
    task_num = 0

    if random_task:
        env.set_random_generate_task(True)
    else:
        env.reset_task_list(task_list)

    now_task = None
    result_done = []
    success = True 

    while True:
        # Env step
        a = policy.get_action(obs, now_task, info)
        a = np.clip(a, -1, 1)
        obs, reward, done, info = env.step(a)

        # Save data
        demo['action'].append(a)
        demo['obs'].append(obs)
        demo['reward'].append(reward)
        demo['src_reward'].append(env.last_reward)
        demo['done'].append(done)

        img_dict = get_img(env)
        for k, v in img_dict.items():
            demo['img'][k].append(v)

        if now_task is None:
            demo = clear_demo(demo)

        now_task = info['task_name']
        task_step = info['task_step']

        if task_step > max_task_step:
            success = False
            break

        if done:
            # Save demo
            save_fail_gif(now_task, seed, task_num, demo, False)
            demo = cal_return_to_go(demo)
            write_h5(now_task, seed, task_num, demo, save_gif=debug)
            result_done.append(now_task)
            task_num += 1
            # Clear demo
            demo = clear_demo(demo)

            if random_task and task_num > total_task_num:
                break
            if not random_task and len(env.task_list) == 0:
                break

        step += 1
    
    if success:
        now_task = None
    
    if not success:
        # save fail task gif
        save_fail_gif(now_task, seed, task_num, demo)

    result_info = {
        'task_list': result_done,
        'fail_task': now_task,
    }
    
    return result_info, success

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

def write_h5(task_name, seed, task_num, demo, save_gif=False):
    root_path = Path(__file__).parent / 'data' / 'display' / task_name
    root_path.mkdir(exist_ok=True, parents=True)
    file_path = root_path / (str(seed) + '-' + str(task_num) + ".hdf5")
    f = h5py.File(file_path, "w")
    for k, v in demo.items():
        if isinstance(v, dict):
            g = f.create_group(k)
            for vk, vv in v.items():
                g.create_dataset(vk, data=vv, compression='gzip')
        else:
            f.create_dataset(k, data=v,compression='gzip')
    f.close()
    if save_gif:
        show_demo('display', str(seed) + '-' + str(task_num) + '-' + task_name, demo, True)

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

# def thread_test_env(task_name, begin_seed, end_seed, q):
#     for seed in range(begin_seed, end_seed):
#         demo, success = run_demo(task_name=task_name, seed=seed)
#         if not success:
#             q.put(seed)

# def test_env(task_name, thread_num=20):
#     thread_len = 100 // thread_num
#     q = multiprocessing.Queue()
#     thread_list = []
#     for t_id in range(thread_num):
#         t = multiprocessing.Process(target=thread_test_env, args=(task_name, t_id*thread_len, (t_id+1)*thread_len, q))
#         t.start()
#         thread_list.append(t)
#     for t in thread_list:
#         t.join()

#     fail_list = []
#     while not q.empty():
#         fail_seed = q.get()
#         fail_list.append(fail_seed)
#     fail_list.sort()
#     print("Fail list:", fail_list)

def thread_generate_data(t_id, task_list, begin_seed, end_seed):
    root_path = Path(__file__).parent / 'data' / 'display'
    root_path.mkdir(exist_ok=True, parents=True)
    success_file_path = root_path / ("fail_" + str(begin_seed) + '-' + str(end_seed) + ".txt")
    for seed in range(begin_seed, end_seed):
        
        result_info, success = data_demo(task_list=task_list, seed=seed)
        if not success:
            with open(success_file_path, "a") as f:
                string = f"{seed} success_task_num: {len(result_info['task_list'])} fail_task_name: {result_info['fail_task']}"
                f.write(string + "\n")

def generate_data(task_list, thread_num=10, begin_seed=0, end_seed=21000):
    thread_list = []
    total_ep = end_seed - begin_seed
    each_thread_ep = total_ep // thread_num
    for t_id in range(thread_num):
        thread_begin_seed = begin_seed + t_id * each_thread_ep
        thread_end_seed = begin_seed + (t_id + 1) * each_thread_ep
        t = multiprocessing.Process(target=thread_generate_data, args=(t_id, task_list, thread_begin_seed, thread_end_seed))
        t.start()
        thread_list.append(t)
    for t in thread_list:
        t.join()

def generate_data_main(random_task=False, begin_seed=0, end_seed=210000):
    if random_task:
        generate_data(task_list=None, thread_num=1, begin_seed=begin_seed, end_seed=end_seed)
    else:
        task_lists = [
            ['coffee-push', 'coffee-button','coffee-pull'],
            ['drawer-open', 'drawer-place', 'drawer-close', 'reset', 
                'drawer-open', 'drawer-pick', 'coffee-push', 'coffee-button',
                'coffee-pull', 'drawer-place', 'drawer-close', 'reset']
            ]
        for task_list in task_lists:
            generate_data(task_list=task_list, thread_num=10, begin_seed=begin_seed, end_seed=end_seed)

def display_random_demo(seed=None, max_task_step=400):
    random.seed(seed)
    np.random.seed(seed)
    env = TASK_DICK['display']['env'](seed=seed)
    policy = TASK_DICK['display']['policy']()

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
    step = 0
    info = {}
    
    task_step = 0
    task_num = 0

    env.set_random_generate_task(True)
    now_task = None
    result_done = []
    success = True 

    while task_num < 20:
        a = policy.get_action(obs, now_task, info)
        a = np.clip(a, -1, 1)

        demo['action'].append(a)
        obs, reward, done, info = env.step(a)
        now_task = info['task_name']
        task_step = info['task_step']

        if task_step > max_task_step:
            success = False
            break

        if done:
            result_done.append(now_task)
            task_num += 1

        demo['obs'].append(obs)
        demo['reward'].append(reward)
        demo['src_reward'].append(env.last_reward)
        demo['done'].append(done)

        img_dict = get_img(env)
        for k, v in img_dict.items():
            demo['img'][k].append(v)

        step += 1
        
    print(f"{seed} {step} {success} {len(result_done)}: {result_done}")
    demo['task_list'] = result_done
    if not success:
        demo['fail_task'] = now_task
    
    return demo, success

def test_display(seed_range=[0, 20]):
    failed_seeds = []
    for seed in range(*seed_range):
        # if seed != 13:
        #     continue
        demo, success = display_random_demo(seed)
        if not success:
            failed_seeds.append(seed)
            show_demo(task_name='display', seed=seed, demo=demo, gif=True)
    print(f'Failed seeds: {failed_seeds}')


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    # test_display()
    # data_demo(task_list=None, seed=12001, total_task_num=10, debug=True)
    # for seed in range(0, 10):
    #     data_demo(task_list=None, seed=seed, total_task_num=50, debug=False)
    generate_data_main(random_task=True, begin_seed=2, end_seed=20+1)
    # task_name = "display"
    # task_list = ['drawer-open', 'drawer-place', 'drawer-close', 'reset', 
    #                 'drawer-open', 'drawer-pick', 'coffee-push', 'coffee-button',
    #                 'coffee-pull', 'drawer-place', 'drawer-close', 'reset']
    # # task_list = ['coffee-push', 'coffee-button','coffee-pull']
    # # task_list = ['drawer-place']
    # for seed in range(1):
    #     # seed = 6
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     demo, _ = run_demo(task_name=task_name, task_list=copy.deepcopy(task_list), seed=seed, max_step=2000, debug=True)
    #     # demo = cal_return_to_go(demo)
    #     show_demo(task_name=task_name, seed=seed, demo=demo, gif=True)

        # write_h5(task_name=task_name, seed=seed, demo=demo)
    # read_h5(task_name=task_name, seed=seed)

    # test_env(task_name)
    

    
