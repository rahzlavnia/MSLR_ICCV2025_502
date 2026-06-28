import torch
import random
import numpy as np

EPS = 1e-4

SENTENCE_LENGTH_BOUNDS = {
    "S01": {"min_len": 115, "max_len": 245},
    "S02": {"min_len": 179, "max_len": 261},
    "S03": {"min_len": 99, "max_len": 180},
    "S04": {"min_len": 184, "max_len": 323},
    "S05": {"min_len": 172, "max_len": 435},
    "S06": {"min_len": 110, "max_len": 150},
    "S07": {"min_len": 220, "max_len": 365},
    "S08": {"min_len": 159, "max_len": 315},
    "S09": {"min_len": 194, "max_len": 339},
    "S10": {"min_len": 95, "max_len": 185},
    "S11": {"min_len": 110, "max_len": 215},
    "S12": {"min_len": 200, "max_len": 340},
    "S13": {"min_len": 150, "max_len": 290},
    "S14": {"min_len": 133, "max_len": 230},
    "S15": {"min_len": 94, "max_len": 184},
    "S16": {"min_len": 154, "max_len": 320},
    "S17": {"min_len": 185, "max_len": 330},
    "S18": {"min_len": 127, "max_len": 240},
    "S19": {"min_len": 155, "max_len": 335},
    "S20": {"min_len": 96, "max_len": 220},
    "S21": {"min_len": 164, "max_len": 283},
    "S22": {"min_len": 179, "max_len": 329},
    "S23": {"min_len": 110, "max_len": 265},
    "S24": {"min_len": 81, "max_len": 165},
    "S25": {"min_len": 203, "max_len": 450},
    "S26": {"min_len": 183, "max_len": 310},
    "S27": {"min_len": 94, "max_len": 160},
    "S28": {"min_len": 89, "max_len": 189},
    "S29": {"min_len": 145, "max_len": 235},
    "S30": {"min_len": 141, "max_len": 260},
}

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, skeleton, **kwargs):
        for t in self.transforms:
            skeleton = t(skeleton, **kwargs)
        return skeleton


class ToTensor(object):
    def __call__(self, skeleton, **kwargs):
        if isinstance(skeleton, np.ndarray):
            skeleton = np.asarray(skeleton)
            skeleton = torch.from_numpy(skeleton).float()
        elif not torch.is_tensor(skeleton):
            skeleton = torch.tensor(skeleton, dtype=torch.float32)
        return skeleton


class Downsample(object):
    """Standalone  downsampling (decoupled from augmentation).

    Picks every ``step``-th frame, where ``step = round(1 / ratio)``. For the
    default ``ratio=0.5`` this keeps every other frame (step 2), matching the
    legacy ``::2`` behaviour that used to live inside the temporal augmentation
    classes.

    Args:
        ratio (float): Fraction of frames to keep (0.5 -> keep half).
        random_offset (bool): If True, randomly start at frame 0 or 1 (the
            legacy train behaviour). If False, always start at 0 (the legacy
            test behaviour).
    """

    def __init__(self, ratio=0.5, random_offset=True) -> None:
        self.step = max(1, int(round(1.0 / ratio)))
        self.random_offset = random_offset

    def __call__(self, clip, **kwargs):
        if self.random_offset:
            start_idx = 0 if random.uniform(0, 1) > 0.5 else 1
        else:
            start_idx = 0
        index = list(range(start_idx, len(clip), self.step))
        return clip[index]


class Jitter(object):
    """
    Apply Gaussian jitter (noise) to skeleton sequences.

    Args:
        std_dev (float): Standard deviation of the Gaussian noise.
    """

    def __init__(self, std_dev=0.006) -> None:
        self.std_dev = std_dev

    def __call__(self, skeleton, **kwargs):
        noise = np.random.normal(loc=0, scale=self.std_dev, size=skeleton.shape)
        return skeleton + noise

class TemporalDropout(object):
    """
    Apply temporal dropout by randomly removing a contiguous segment of frames.

    NOTE: downsampling is NO LONGER applied here. To reproduce the legacy
    behaviour, follow this transform with ``Downsample(ratio=0.5)``.

    Args:
        max_dp (float): Maximum dropout proportion. Actual dropout length
            is between [0, vid_len * max_dp].
    """

    def __init__(self, max_dp=0.2) -> None:
        self.max_dp = max_dp

    def __call__(self, skeleton, **kwargs):
        vid_len = len(skeleton)
        dp_len = int(vid_len * self.max_dp * np.random.random())
        start = np.random.randint(0, vid_len - dp_len + 1)
        end = start + dp_len
        index = list(range(0, start)) + list(range(end, vid_len))
        return skeleton[index]


class TemporalCrop(object):
    """
    Apply temporal cropping by dropping frames from the beginning and end.

    NOTE: downsampling is NO LONGER applied here. Follow with
    ``Downsample(ratio=0.5)`` to reproduce the legacy behaviour.

    Args:
        max_dp (float): Maximum dropout proportion. Actual dropout length
            is between [0, vid_len * max_dp].
    """

    def __init__(self, max_dp=0.2) -> None:
        self.max_dp = max_dp

    def __call__(self, skeleton, **kwargs):
        vid_len = len(clip)
        dp_len = int(vid_len * self.max_dp * np.random.random())
        drop_head = random.randint(0, dp_len)
        drop_tail = dp_len - drop_head

        head_idx = drop_head
        end_idx = vid_len - drop_tail
        index = list(range(head_idx, end_idx))
        return clip[index]


class Dropout_kp(object):
    """
    Apply dropout to skeleton keypoints.

    Args:
        drop_prob (float): Probability of dropping each keypoint at each frame.
    """

    def __init__(self, drop_prob=0.1) -> None:
        self.drop_prob = drop_prob

    def __call__(self, skeleton, **kwargs):
        T, K, _ = skeleton.shape
        mask = np.random.rand(T, K) > self.drop_prob
        return skeleton * mask[..., np.newaxis]


class Spatial_flip(object):
    """
    Apply spatial flipping to skeleton sequences.

    Args:
        prob (float): Probability of applying spatial flip.
    """

    def __init__(self, prob=0.5) -> None:
        self.prob = prob

    def __call__(self, skeleton):
        flag = random.random() < self.prob
        if flag:
            flipped_skeleton = skeleton.copy()
            flipped_skeleton[..., 0] = - flipped_skeleton[..., 0]
            flipped_skeleton[:, 0:21], flipped_skeleton[:, 21:42] = flipped_skeleton[:, 21:42].copy(), flipped_skeleton[:, 0:21].copy()
            return flipped_skeleton
        else:
            return skeleton


class Scale(object):
    """
    Scale skeleton sequences by applying random scaling factors.

    Args:
        scale_range (tuple): Range of scaling factors (min, max).
    """

    def __init__(self, scale_range=(0.8, 1.2)) -> None:
        self.scale_range = scale_range

    def __call__(self, skeleton, **kwargs):
        T = skeleton.shape[0]
        scales = np.random.uniform(*self.scale_range, size=T)
        scaled_skeleton = skeleton * scales[:, np.newaxis, np.newaxis]
        return scaled_skeleton


class TemporalRescale(object):
    """
    Temporally rescale video by resampling frames.

    NOTE: downsampling is NO LONGER applied here. Follow with
    ``Downsample(ratio=0.5)`` to reproduce the legacy behaviour.

    Args:
        temp_scaling (float): Temporal scaling factor. Video length is scaled
            between [1 - temp_scaling, 1 + temp_scaling].
        bounds (dict): Optional per-sentence {sentence_id: {min_len, max_len}}
            override. Defaults to SENTENCE_LENGTH_BOUNDS.
        default_min_len/default_max_len: Fallback bounds when sentence_id is
            unknown or not provided.
    """

    def __init__(self, temp_scaling=0.2, bounds=None, default_min_len=32, default_max_len=230) -> None:
        self.bounds = bounds if bounds is not None else SENTENCE_LENGTH_BOUNDS
        self.default_min_len = default_min_len
        self.default_max_len = default_max_len
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip, sentence_id=None, **kwargs):
        # clip shape: T X N X 2
        b = self.bounds.get(sentence_id, {})
        min_len = b.get("min_len", self.default_min_len)
        max_len = b.get("max_len", self.default_max_len)

        vid_len = len(clip)
        new_len = int(vid_len * np.random.uniform(self.L, self.U))
        if new_len < min_len:
            new_len = min_len
        if new_len > max_len:
            new_len = max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]


class TemporalRescale_test(object):
    """
    Deterministic test-time temporal padding.

    NOTE: downsampling is NO LONGER applied here. Follow with
    ``Downsample(ratio=0.5, random_offset=False)`` to reproduce the legacy
    test behaviour.
    """

    def __call__(self, clip, **kwargs):
        # clip shape: T X N X 2
        vid_len = len(clip)
        new_len = vid_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        index = [i for i in range(new_len)]
        for i in range(vid_len, new_len):
            index[i] = index[vid_len - 1]
        return clip[index]