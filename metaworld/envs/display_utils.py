import numpy as np
import matplotlib.colors

COLOR_LIST = [
    "#F27970",
    "#BB9727",
    "#54B345",
    "#32B897",
    "#05B9E2",
    "#8983BF",
    "#C76DA2",
]
RGB_COLOR_LIST = [matplotlib.colors.to_rgb(color) for color in COLOR_LIST]

QUAT_LIST = [
    [1., 0., 0., 0.],                               # 正常状态
    [np.sin(np.pi/4), 0., 0., np.sin(np.pi/4)],     # 俯视顺时针旋转90
    [np.sin(np.pi/4), 0., 0., -np.sin(np.pi/4)],    # 俯视逆时针旋转90
    [0., 0., 0., 1.],                               # 俯视旋转180
]