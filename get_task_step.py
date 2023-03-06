from pathlib import Path
import os

BIN_PATH = Path(__file__).absolute().parents[0] / "data_bin"
# BIN_PATH = Path("/home/data") / "metaworld-bin"

def get_steps(task):
    length = 0
    task_path = BIN_PATH / task
    for d in os.listdir(task_path):
        file_path = task_path / d / "length.txt"
        with open(file_path, "r") as f:
            length_i = f.read()
            length += int(length_i)
    print(f"{task}: {length}")

task_list = [
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
    "reach",
    "reach-wall",
    "shelf-place",
    # "soccer",
    "stick-pull",
    "stick-push",
    "sweep-into",
    "sweep",
    "window-close",
    "window-open",
]

for task in task_list:
    get_steps(task)
