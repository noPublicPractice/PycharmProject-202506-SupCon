import numpy as np
import torch
import random

# https://arxiv.org/pdf/1706.00527.pdf
def DataTransform(sample, config):
    weak_aug = scaling(sample, sigma=config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    return weak_aug, strong_aug
def jitter(x, sigma=0.8):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)  # 正态分布的
def scaling(x, sigma=1.1):
    x = x.cpu().numpy()  # 为什么在cpu
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))  # 正态分布的
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])  # x-3D，
    # 笔记：GPU中的Variable变量：a.cuda().data.cpu().numpy()
    # 笔记：GPU中的tensor变量：a.cuda().cpu().numpy()
    # 笔记：numpy不能直接读取CUDA tensor，需要将它转化为CPU tensor。把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。
    return np.concatenate((ai), axis=1)
def permutation(x, max_segments=5, seg_mode="random"):  # 时间序列分段再打乱
    orig_steps = np.arange(x.shape[2])
    x = x.cpu().numpy()
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            # a = list(range(len(splits)))
            # random.shuffle(a)
            # temp_list = []
            # for k in a:
            #     temp_list.extend(list(splits[k]))
            # warp = np.array(temp_list)
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)
