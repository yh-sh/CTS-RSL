import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
# class SpatialTransform(nn.Module):
#     def __init__(self, seg_size, max_weight = 1.5, max_bias = 0.1):
#         super(SpatialTransform, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(torch.zeros(seg_size)))
#         self.bias = nn.Parameter(torch.Tensor(torch.zeros(seg_size)))
#         self.max_weight = max_weight
#         self.max_bias = max_bias
#     def forward(self, input, comparison):
#         output = input * self.max_weight * (torch.sigmoid(self.weight)+0.5)  + self.max_bias * torch.tanh(self.bias)
#         return output

class SpatialTransform(nn.Module):
    def __init__(self, configs):
        super(SpatialTransform, self).__init__()

        self.conv1 = nn.Conv1d(configs.input_channels, configs.channel, kernel_size=configs.kernel, padding=configs.padding)
        self.conv2 = nn.Conv1d(configs.channel, 2, kernel_size=configs.kernel,  padding=configs.padding)

    def forward(self, input):
        # Apply convolutional layers
        x = torch.relu(self.conv1(input))
        transform_factors = torch.sigmoid(self.conv2(x))

        jitter_factors = transform_factors[:, 0, :] - 0.5
        scale_factors = transform_factors[:, 1, :] + 0.5

        # Apply jitter and scale
        # Assuming you want to apply the same transformation across all channels
        jittered_and_scaled = input * scale_factors.unsqueeze(1) + jitter_factors.unsqueeze(1)

        return jittered_and_scaled


def DataTransform(sample, config):
    temporal_aug = TemporalTransform(jitter(sample, config.augmentation.jitter_ratio))
    spatial_aug = sample
    

    return temporal_aug, spatial_aug

def CDataTransform(sample, config, ablation):
    #Par, Par_Per, Par_Sca, Par_Jit, Tem, Tem_Per, Tem_Sca, Tem_Jit, Sca_Jit, Per_Jit, Per_Sca
    if ablation == "Par":
        aug1 = sample
        aug2 = sample
    elif ablation == "Tem":
        aug1 = TemporalTransform(jitter(sample, config.augmentation.jitter_ratio))
        aug2 = sample
    elif ablation == "Par_Per":
        aug1 = permutation(sample, max_segments=config.augmentation.max_seg)
        aug2 = sample
    elif ablation == "Par_Sca":
        aug1 = scaling(sample, sigma=config.augmentation.jitter_scale_ratio)
        aug2 = sample
    elif ablation == "Par_Jit":
        aug1 = jitter(sample, config.augmentation.jitter_ratio)
        aug2 = sample
    elif ablation == "Tem_Per":
        aug1 = TemporalTransform(jitter(sample, config.augmentation.jitter_ratio))
        aug2 = permutation(sample, max_segments=config.augmentation.max_seg)
    elif ablation == "Tem_Sca":
        aug1 = TemporalTransform(jitter(sample, config.augmentation.jitter_ratio))
        aug2 = scaling(sample, sigma=config.augmentation.jitter_scale_ratio)
    elif ablation == "Tem_Jit":
        aug1 = TemporalTransform(jitter(sample, config.augmentation.jitter_ratio))
        aug2 = jitter(sample, config.augmentation.jitter_ratio)
    elif ablation == "Sca_Jit":
        aug1 = scaling(sample, sigma=config.augmentation.jitter_scale_ratio)
        aug2 = jitter(sample, config.augmentation.jitter_ratio)
    elif ablation == "Per_Jit":
        aug1 = permutation(sample, max_segments=config.augmentation.max_seg)
        aug2 = jitter(sample, config.augmentation.jitter_ratio)
    elif ablation == "Per_Sca":
        aug1 = permutation(sample, max_segments=config.augmentation.max_seg)
        aug2 = scaling(sample, sigma=config.augmentation.jitter_scale_ratio)
    
    return aug1, aug2


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    x = x.cpu().numpy()
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)

def TemporalTransform(x, r=0.3, expansion_factor=2):
    n = x.shape[2]
    ret = np.zeros_like(x)
    segment_len = int(r*n)
    starts = np.random.randint(0, n - segment_len, size=(x.shape[0]))

    for i, pat in enumerate(x):
        expand_start = starts[i]
        expand_end = expand_start + segment_len
        orig_indices = np.arange(expand_start, expand_end)
        expand_indices = np.linspace(expand_start, expand_end, int((expand_end - expand_start) * expansion_factor), endpoint=False)
        for series in pat:
            expanded_section = interp1d(orig_indices, series[expand_start:expand_end], kind='linear', fill_value='extrapolate')(expand_indices)
            expanded_series = np.concatenate([series[:expand_start], expanded_section, series[expand_start+segment_len:]])
            reshape_indices = np.linspace(0, len(expanded_series), n, endpoint=False)
            adjusted_series = interp1d(np.arange(len(expanded_series)), expanded_series, kind='linear', fill_value='extrapolate')(reshape_indices)
        ret[i] = adjusted_series
    return torch.from_numpy(ret)

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2]) #timeline
    x = x.cpu().numpy()
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0])) #(data_num) * rand(1-5)

    ret = np.zeros_like(x) 
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False) # for range(0, x.shape[2]-2) select num_segs[i]-1 samples without replacement
                split_points.sort()
                splits = np.split(orig_steps, split_points) # split with points
            else:
                splits = np.array_split(orig_steps, num_segs[i]) # split with fix sections
            warp = np.concatenate(np.random.permutation(splits)).ravel() # permutate splited sections
            ret[i] = pat[:, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)
