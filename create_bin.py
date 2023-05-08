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

import tensorflow as tf

IMG_ENCODE_FORMAT = 'jpg'

ROOT_PATH = Path(__file__).absolute().parents[0] / "data"
BIN_PATH = Path(__file__).absolute().parents[0] / "data_bin"
TASK_LIST = [
        # "assembly",
        # "basketball",
        # "bin-picking",
        # "box-close",
        # "button-press-topdown",
        # "button-press-topdown-wall",
        # "button-press",
        # "button-press-wall",
        # "coffee-button",
        # "coffee-pull",
        # "coffee-push",
        # "dial-turn",
        # "disassemble",
        # "door-close",
        # "door-lock",
        # "door-open",
        # "door-unlock",
        # "drawer-close",
        # "drawer-open",
        # "faucet-close",
        # "faucet-open",
        # "hammer",
        # "hand-insert",
        # "handle-press-side",
        # "handle-press",
        # "handle-pull-side",
        # "handle-pull",
        # "lever-pull",
        # "peg-insert-side",
        # "peg-unplug-side",
        # "pick-out-of-hole",
        # "pick-place",
        # "pick-place-wall",
        # "plate-slide-back-side",
        # "plate-slide-back",
        # "plate-slide-side",
        # "plate-slide",
        # "push-back",
        # "push",
        # "push-wall",
        # "reach",
        # "reach-wall",
        # "shelf-place",
        # "soccer",
        # "stick-pull",
        # "stick-push",
        # "sweep-into",
        # "sweep",
        # "window-close",
        # "window-open",
        # "coffee-button-display",
        # "coffee-pull-display",
        # "coffee-push-display",
        "drawer-close-display",
        "drawer-open-display",
        "drawer-pick-display",
        "drawer-place-display", 
        ]
EPISODE_LIST = [i for i in range(21000)]
# REPLAY_LIST = [str(i) for i in range(1, 2)]
CKPT_LIST = [i for i in range(210)]

THREAD_NUM = 50
PROCESS_NUM = 10

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
    for i in range(id_begin, id_end):
        src_path = ROOT_PATH / task / ("%d.hdf5" % i)
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
    for i in range(length):
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
    for i in tqdm(range(length)):
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

    begin_episode = ckpt * (len(EPISODE_LIST) // len(CKPT_LIST))

    data_thread_dict = {}
    valid_len = len(EPISODE_LIST) // len(CKPT_LIST)
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
        for ckpt in CKPT_LIST:
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

def random_test(test_num=1000, test_in_num=10):
    ckpt_episode_len = len(EPISODE_LIST) // len(CKPT_LIST)
    for _ in range(test_num):
        task = random.choice(TASK_LIST)
        ckpt = random.choice(CKPT_LIST)
        # ckpt = 0
        
        length_path = BIN_PATH / task / str(ckpt) / "length.txt"
        data_bin_path = BIN_PATH / task / str(ckpt) / "data.bin"
        index_bin_path = BIN_PATH / task / str(ckpt) / "index.bin"
        
        file_exist = check_file(length_path, )
        if not file_exist:
            print("Skip")
            continue

        data_f = open(data_bin_path, 'rb')
        data_index_f = open(index_bin_path, 'rb')

        length_f = open(length_path, 'r')
        valid_len = int(length_f.read())
        length_f.close()

        for _j in range(test_in_num):
            test_lenth = 4
            index = random.randint(0, valid_len-test_lenth-1)

            data_index_f.seek(index * 4 * 8)
            data_index = np.frombuffer(data_index_f.read(test_lenth * 4 * 8), dtype=np.int64).reshape(-1, 4)

            for i, (data_offset, img1_len, img2_len, img3_len) in enumerate(data_index):
                data_f.seek(data_offset)

                save_act = np.frombuffer(data_f.read(4*4), dtype=np.float32)
                save_rew = np.frombuffer(data_f.read(4), dtype=np.float32)
                save_ter = np.frombuffer(data_f.read(1), dtype=np.uint8)
                save_ret = np.frombuffer(data_f.read(4), dtype=np.float32)
                save_obs = np.frombuffer(data_f.read(4*39), dtype=np.float32)

                img1_encode = np.frombuffer(data_f.read(img1_len), dtype=np.uint8)
                save_img1 = cv2.imdecode(img1_encode, -1)
                img2_encode = np.frombuffer(data_f.read(img2_len), dtype=np.uint8)
                save_img2 = cv2.imdecode(img2_encode, -1)
                img3_encode = np.frombuffer(data_f.read(img3_len), dtype=np.uint8)
                save_img3 = cv2.imdecode(img3_encode, -1)

                begin_id = 0
                for j in range(ckpt_episode_len * ckpt, ckpt_episode_len * (ckpt+1)):
                    real_path = ROOT_PATH / task / ("%d.hdf5" % j)
                    if not check_file(real_path, ):
                        continue
                    f = h5py.File(real_path, "r")
                    end_id = begin_id + f["action"].shape[0]
                    if end_id <= index + i:
                        begin_id = end_id
                        f.close()
                        continue

                    h5_id = index + i - begin_id
                    real_act = np.array(f["action"][h5_id], dtype=np.float32)
                    real_rew = np.array(f["reward"][h5_id], dtype=np.float32)
                    real_ter = np.array(f["done"][h5_id], dtype=np.uint8)
                    real_ret = np.array(f["return"][h5_id], dtype=np.float32)
                    real_obs = np.array(f["obs"][h5_id], dtype=np.float32)
                    real_img1 = np.array(f["img"]["corner3"][h5_id], dtype=np.uint8)
                    real_img2 = np.array(f["img"]["corner"][h5_id], dtype=np.uint8)
                    real_img3 = np.array(f["img"]["topview"][h5_id], dtype=np.uint8)
                    f.close()
                    break

                act_ans = (real_act == save_act).all()
                rew_ans = (real_rew == save_rew).all()
                ter_ans = (real_ter == save_ter).all()
                ret_ans = (real_ret == save_ret).all()
                obs_ans = (real_obs == save_obs).all()
                img1_ans = (real_img1 == save_img1).all()
                img2_ans = (real_img2 == save_img2).all()
                img3_ans = (real_img3 == save_img3).all()
                print(f"{task}/{ckpt}/[{index+i}]: act:{act_ans}, rew:{rew_ans}, ter:{ter_ans}, ret:{ret_ans}, obs:{obs_ans}, img:{img1_ans} {img2_ans} {img3_ans}")

                if not (act_ans and obs_ans and rew_ans and ter_ans and ret_ans and img1_ans and img2_ans and img3_ans):
                    raise ValueError(f"Check fail, data not match in {task}/{ckpt}/[{index+i}]")
        
        data_f.close()
        data_index_f.close()
                
def test_single_data(task, file_name):
    act_list, rew_list, ter_list, ret_list, obs_list = [], [], [], [], []
    img1_list, img2_list, img3_list = [], [], []
    
    src_path = ROOT_PATH / task / file_name
    
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
    for i in range(length):
        for img in [img1[i], img2[i], img3[i]]:
            param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
            # img_encode = cv2.imencode('.png', img, param)[1]
            img_encode = cv2.imencode('.jpg', img, param)[1]
            # img_encode = np.frombuffer(tf.io.encode_jpeg(img).numpy(), dtype=np.uint8)
            img_all += len(img_encode)
    
    data_all = img_all + (4*4 + 4 + 1 + 4 + 4*39) * length

    data = np.zeros([data_all], dtype=np.uint8)
    data_index = np.zeros([length, 4], dtype=np.int64)

    act, rew, ter, ret, obs = np2uint8(act, rew, ter, ret, obs)

    data_begin = 0
    for i in tqdm(range(length)):
        param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        # img1_encode = cv2.imencode('.png', img1[i])[1]
        img1_encode = cv2.imencode('.jpg', img1[i], param)[1]
        # img1_decode = cv2.imdecode(img1_encode, -1)
        # print("same:", (img1_decode == img1[i]).all())
        # raise ValueError("")
        # img1_encode = np.frombuffer(tf.io.encode_jpeg(img1[i]).numpy(), dtype=np.uint8)
        img1_encode_len = len(img1_encode)
        # img2_encode = cv2.imencode('.png', img2[i])[1]
        img2_encode = cv2.imencode('.jpg', img2[i], param)[1]
        # img2_encode = np.frombuffer(tf.io.encode_jpeg(img2[i]).numpy(), dtype=np.uint8)
        img2_encode_len = len(img2_encode)
        # img3_encode = cv2.imencode('.png', img3[i])[1]
        img3_encode = cv2.imencode('.jpg', img3[i], param)[1]
        # img3_encode = np.frombuffer(tf.io.encode_jpeg(img3[i]).numpy(), dtype=np.uint8)
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

    bin_dir = BIN_PATH / task
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)
    data_bin_path = bin_dir / "data_640_480_jpg_100.bin"
    data_f = open(data_bin_path, 'wb')
    data_f.write(data_byte)
    data_f.close()


# test_single_data(task='drawer-place-display', file_name='0_640_480.hdf5')
# test_encode_len(task='drawer-place-display', file_name='0_640_480.hdf5')
# write_bin("push-back", 6)
multiprocess_write_bin(PROCESS_NUM)
# random_test(1000)