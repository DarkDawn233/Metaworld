import numpy as np
import os
import gzip
from pathlib import Path
import multiprocessing
import threading
import cv2
from tqdm import tqdm
import copy
import time
import random
import h5py
import math

# import tensorflow as tf

IMG_ENCODE_FORMAT = 'jpg'

ROOT_PATH = Path(__file__).absolute().parents[0] / "data" / "display"
BIN_PATH = Path(__file__).absolute().parents[0] / "data_bin"
TASK_LIST = [
        "bin-pick",
        "bin-place"
        "coffee-button",
        "coffee-pull",
        "coffee-push",
        "desk-pick",
        # "desk-place",
        # "drawer-close",
        # "drawer-open",
        # "drawer-pick",
        # "drawer-place",
        # "reset"
        ]

THREAD_NUM = 50
PROCESS_NUM = 5
EPISODE_NUM = 200

def check_file(*paths):
    f = True
    for path in paths:
        if not os.path.exists(path):
            print("Not exist", path)
            f = False
    return f

def clear_file(*paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)

def np2byte(*args):
    ans = []
    for arg in args:
        ans.append(arg.tobytes())
    return ans

def np2uint8(*args):
    ans = []
    for arg in args:
        tmp = arg.tobytes()
        ans.append(np.frombuffer(tmp, dtype=np.uint8))
    return ans

def thread_process_data(t_id, task, id_begin, id_end, t_dict, img_encode_format='png'):
    act_list, rew_list, ter_list, ret_list, obs_list = [], [], [], [], []
    img1_list, img2_list, img3_list = [], [], []
    task_root_path = ROOT_PATH / task
    task_file_list = os.listdir(str(task_root_path))
    task_file_list.sort()
    for i in range(id_begin, id_end):
        src_path = task_root_path / task_file_list[i]
        print(f"read file {i-id_begin} {id_end-id_begin} {src_path}")
        if not check_file(src_path, ):
            continue
        
        f = h5py.File(src_path, "r")
        act_list.append(np.array(f["action"]))
        rew_list.append(np.array(f["reward"]))
        ter_list.append(np.array(f["done"]))
        ret_list.append(np.array(f["return"]))
        obs_list.append(np.array(f["obs"])[:-1])

        img1_list.append(np.array(f["img"]["corner3"])[:-1])
        img2_list.append(np.array(f["img"]["corner"])[:-1])
        img3_list.append(np.array(f["img"]["topview"])[:-1])

        f.close()
        # print(f"Thread {t_id} file {i} read finish.")
    
    if len(act_list) == 0:
        print(f"Thread {t_id} has no data, finish.")
        return

    act = np.array(np.concatenate(act_list, axis=0), dtype=np.float32)
    rew = np.array(np.concatenate(rew_list, axis=0), dtype=np.float32)
    ter = np.array(np.concatenate(ter_list, axis=0), dtype=np.uint8)
    ret = np.array(np.concatenate(ret_list, axis=0), dtype=np.float32)
    obs = np.array(np.concatenate(obs_list, axis=0), dtype=np.float32)
    img1 = np.array(np.concatenate(img1_list, axis=0), dtype=np.uint8)
    img2 = np.array(np.concatenate(img2_list, axis=0), dtype=np.uint8)
    img3 = np.array(np.concatenate(img3_list, axis=0), dtype=np.uint8)

    length = act.shape[0]

    img_all = 0
    for i in tqdm(range(length), desc=task + str(t_id) + 'get_lenth'):
        for img in [img1[i], img2[i], img3[i]]:
            if img_encode_format == "jpg":
                img_encode = cv2.imencode('.jpg', img)[1]
            else:
                img_encode = cv2.imencode('.png', img)[1]
            img_all += len(img_encode)
    
    data_all = img_all + (4*4 + 4 + 1 + 4 + 4*39) * length

    data = np.zeros([data_all], dtype=np.uint8)
    data_index = np.zeros([length, 4], dtype=np.int64)

    act, rew, ter, ret, obs = np2uint8(act, rew, ter, ret, obs)

    data_begin = 0
    for i in tqdm(range(length), desc=task + str(t_id) + 'get_data'):
        if img_encode_format == "jpg":
            img1_encode = cv2.imencode('.jpg', img1[i])[1]
        else:
            img1_encode = cv2.imencode('.png', img1[i])[1]
        img1_encode_len = len(img1_encode)
        if img_encode_format == "jpg":
            img2_encode = cv2.imencode('.jpg', img2[i])[1]
        else:
            img2_encode = cv2.imencode('.png', img2[i])[1]
        img2_encode_len = len(img2_encode)
        if img_encode_format == "jpg":
            img3_encode = cv2.imencode('.jpg', img3[i])[1]
        else:
            img3_encode = cv2.imencode('.png', img3[i])[1]
        img3_encode_len = len(img3_encode)

        tmp = data_begin
        data[tmp: tmp + 4*4] = act[i * 4*4: (i+1) * 4*4]
        tmp += 4*4
        data[tmp: tmp + 4] = rew[i * 4: (i+1) * 4]
        tmp += 4
        data[tmp: tmp + 1] = ter[i * 1: (i+1) * 1]
        tmp += 1
        data[tmp: tmp + 4] = ret[i * 4: (i+1) * 4]
        tmp += 4
        data[tmp: tmp + 4*39] = obs[i * 4*39: (i+1) * 4*39]
        tmp += 4*39
        data[tmp: tmp + img1_encode_len] = img1_encode
        tmp += img1_encode_len
        data[tmp: tmp + img2_encode_len] = img2_encode
        tmp += img2_encode_len
        data[tmp: tmp + img3_encode_len] = img3_encode
        tmp += img3_encode_len

        data_index[i] = [data_begin, img1_encode_len, img2_encode_len, img3_encode_len]
        data_begin += 4*4 + 4 + 1 + 4 + 4*39 + img1_encode_len + img2_encode_len + img3_encode_len

    data_byte = data.tobytes()
    t_dict[t_id] = {"data_byte": data_byte, "data_index": data_index, "data_length": length}
    print(f"Thread {t_id} finish.")

def write_bin(task, ckpt):

    begin_episode = ckpt * EPISODE_NUM

    data_thread_dict = {}
    valid_len = EPISODE_NUM
    thread_len = valid_len // THREAD_NUM
    thread_list = []
    for i in range(THREAD_NUM):
        thread_begin = i*thread_len + begin_episode
        if i == THREAD_NUM - 1:
            thread_end = max((i+1)*thread_len, valid_len) + begin_episode
        else:
            thread_end = (i+1)*thread_len + begin_episode
        t = threading.Thread(target=thread_process_data, args=(i, task, thread_begin, thread_end, data_thread_dict, IMG_ENCODE_FORMAT))
        t.start()
        thread_list.append(t)
    for t in thread_list:
        t.join()
    
    data_byte = b""
    data_index_list = []
    data_begin = np.array([0, 0, 0, 0], dtype=np.int64)
    data_length = 0
    for i in range(THREAD_NUM):
        if i not in data_thread_dict:
            continue
        data_byte += data_thread_dict[i]["data_byte"]
        data_index_list.append(data_thread_dict[i]["data_index"]+data_begin)
        data_begin[0] += data_thread_dict[i]["data_index"][-1].sum() + 4*4 + 4 + 1 + 4 + 4*39
        data_length += data_thread_dict[i]["data_length"]
    
    if len(data_index_list) == 0:
        print(f"No data ckpt finish: {task} {ckpt}")
        return

    data_index = np.concatenate(data_index_list, axis=0)
    data_index_byte = data_index.tobytes()

    bin_dir = BIN_PATH / task / str(ckpt)
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    data_bin_path = bin_dir / "data.bin"
    index_bin_path = bin_dir / "index.bin"
    clear_file(data_bin_path, index_bin_path)

    data_f = open(data_bin_path, 'wb')
    index_f = open(index_bin_path, 'wb')
    data_f.write(data_byte)
    index_f.write(data_index_byte)

    length_f = open(bin_dir / "length.txt", 'w')
    length_f.write(str(data_length))

    data_f.close()
    index_f.close()
    length_f.close()

    print(f"write ckpt finish: {task} {ckpt}")

def multiprocess_write_bin(process_num = 20):
    process_list = []
    for task in TASK_LIST:
        task_root_path = ROOT_PATH / task
        if not check_file(task_root_path, ):
            continue
        task_data_num = len(os.listdir(str(task_root_path)))
        ckpt_num = math.ceil(task_data_num / EPISODE_NUM)
        for ckpt in range(ckpt_num):
            while len(process_list) == process_num:
                process_done = False
                for i, p in enumerate(process_list):
                    if not p.is_alive():
                        process_done = True
                        process_list.pop(i)
                        break
                if process_done:
                    break
                else:
                    time.sleep(60)
            p = multiprocessing.Process(target=write_bin, args=(task, ckpt, ))
            p.start()
            process_list.append(p)


# test_single_data(task='drawer-place-display', file_name='0_640_480.hdf5')
# test_encode_len(task='drawer-place-display', file_name='0_640_480.hdf5')
# write_bin("desk-place", 0)
multiprocess_write_bin(PROCESS_NUM)
# random_test(1000)
