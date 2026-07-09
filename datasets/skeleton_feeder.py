import os
import sys
import pdb
import json
import torch
import pickle
import warnings
import itertools
import random

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import torch.utils.data as data
from utils import skeleton_augmentation
from itertools import chain

sys.path.append("..")

class SkeletonFeeder(data.Dataset):

    AUGMENTATION_REGISTRY = {
        "TemporalDropout": skeleton_augmentation.TemporalDropout,
        "TemporalCrop":    skeleton_augmentation.TemporalCrop,
        "TemporalRescale": skeleton_augmentation.TemporalRescale,
        "Jitter":          skeleton_augmentation.Jitter,
        "Scale":           skeleton_augmentation.Scale,
        "Dropout_kp":      skeleton_augmentation.Dropout_kp,
        "Spatial_flip":    skeleton_augmentation.Spatial_flip,
    }

    def __init__(
        self,
        gloss_dict,
        mode="train",
        setting="si",
        transform_mode=True,
        datatype="lmdb",
        dataset='phoenix14',
        si_signer=None,
        split=None,
        norm_point=None,
        used_part=None,
        dataset_root=None,
        augmentation_types=None,
        downsampling=False,
        downsampling_position="after",
        downsampling_ratio=0.5,
    ):
        self.mode = mode
        self.mode_list = mode.split("_")
        self.dict = gloss_dict
        self.setting = setting
        self.data_type = datatype
        self.transform_mode = "train" if transform_mode else "test"
        self.dataset = dataset
        self.used_part = used_part
        if mode in ['train', 'dev']:
            pkl_file = "./datasets/pose_bisindo_train_dev_si.pkl"
            info_file = f"./datasets/mslr2025/{mode}_info.json"
        elif mode == 'test_si_major':
            pkl_file = "./datasets/pose_bisindo_test_si-maj.pkl"
            info_file = f"./datasets/mslr2025/test_si_major_info.json"
        elif mode == 'test_si_minor':
            pkl_file = "./datasets/pose_bisindo_test_si-min.pkl"
            info_file = f"./datasets/mslr2025/test_si_minor_info.json"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        with open(info_file, 'r') as f:
            inputs_list = json.load(f)
            
        with open(pkl_file, "rb") as f:
            self.kps_global = pickle.load(f)

        self.inputs_list = list()
        for item in inputs_list:
            if item['video_id'] in self.kps_global.keys():
                self.inputs_list.append(item)
            else:
                print(item)
        self.norm_div = 0.5 # (proposed value)
        # self.norm_div = (10240 - 1) / 2 (baseline value)
        print(mode, len(self))

        if self.data_type == 'skeleton':
            self.pose_idx = []
            for part in self.used_part:
                if part == 'body':
                    self.pose_idx += [i for i in range(61, 86)]
                elif part == 'hand21':
                    self.pose_idx += [i for i in range(0, 21)]
                    self.pose_idx += [i for i in range(21, 42)]
                elif part == 'mouth_8':
                    self.pose_idx += [i for i in range(42, 61)]
                elif part == 'left_hand':
                    self.pose_idx += [i for i in range(0, 21)]
                elif part == 'right_hand':
                    self.pose_idx += [i for i in range(21, 42)]

        self.split = split
        self.norm_point = norm_point
        if norm_point is None:
            print('no centeralization')

        self.augmentation_types = augmentation_types if augmentation_types is not None else []
        self.downsampling = downsampling
        self.downsampling_ratio = downsampling_ratio
        self.downsampling_position = downsampling_position
        assert self.downsampling_position in ("before", "after"), (
            f"downsampling_position harus 'before'/'after', dapat: {downsampling_position}"
        )

        self.data_aug = self.pose_transform()
    
    def __getitem__(self, idx):
        if self.data_type == 'skeleton':
            input_data, label, fi = self.read_pose(idx)
            sentence_id = fi.get("sentence_id") if isinstance(fi, dict) else None
            input_data = input_data[:, self.pose_idx, :2]
            conf = np.zeros_like(input_data)[:, :, 0]

            total_motion = np.zeros(input_data.shape[0:2] + (4,))
            total_motion[1:, :, 0:2] = input_data[1:, :, 0:2] - input_data[0:-1, :, 0:2]
            total_motion[0:-1, :, 2:4] = input_data[:-1, :, 0:2] - input_data[1:, :, 0:2]

            # T * 79 * 6 (2+4)
            final = np.concatenate([input_data, total_motion, conf[:,:,None]], axis=-1)

            input_data = self.normalize(final, sentence_id=sentence_id)
            if self.mode == 'test':
                return (
                    input_data,
                    torch.LongTensor(label),
                    str(self.inputs_list[idx]),
                )
            else:
                return (
                    input_data,
                    torch.LongTensor(label),
                    self.inputs_list[idx]['original_info'],
                )

    def deleteInvalidInputs(self):
        new_list = []
        for index in range(len(self.inputs_list)-1):
            fi = self.inputs_list[index]
            signer = fi['signer']
            if not signer == 'Signer05':
                new_list.append(fi)
        new_list.append(self.inputs_list['prefix'])
        return new_list

    def read_pose(self, index, num_glosses=-1):
        # load file info
        if self.mode == 'test':
            pose_data = self.kps_global[self.inputs_list[index]]['keypoints']
            label_list = 1
            fi = '[EMPTY]'
        else:
            fi = self.inputs_list[index]
            pose_data = self.kps_global[fi['video_id']]['keypoints']
            label = fi['gloss_sequence']
            label_list = []
            for phase in label.split(" "):
                if phase == '':
                    continue
                if phase in self.dict.keys():
                    label_list.append(self.dict[phase])
        return (
            pose_data,
            label_list,
            fi,
        )

    def normalize(self, video, sentence_id=None, label=None, file_id=None):
        if self.data_type == 'skeleton':
            input_data = self.data_aug(video, sentence_id=sentence_id)
            input_data = self.simple_normalize(input_data)
            return input_data

    def simple_normalize(self, origin_input_data):
        conf = origin_input_data[:,:,6]
        origin_input_data = origin_input_data / self.norm_div - 1

        input_data = origin_input_data[:, :, 0:2]
        if self.norm_point is not None:
            index = 0
            for part in self.used_part:
                if index == 0:
                    start, end = 0, self.split[0]
                else:
                    start, end = self.split[index-1], self.split[index]
                if part == 'body':
                    input_data[:, start:end] = (
                        input_data[:, start:end] - input_data[0,self.norm_point[index]:self.norm_point[index]+2].mean(0)[None,None]
                    )
                elif part == 'hand21':
                    input_data[:, start:end] = (
                        input_data[:, start:end] - input_data[:,self.norm_point[index]][:,None,:]
                    )
                    index += 1
                    start, end = self.split[index-1], self.split[index]
                    input_data[:, start:end] = (
                        input_data[:, start:end] - input_data[:,self.norm_point[index]][:,None,:]
                    )
                else:
                    input_data[:, start:end] = (
                        input_data[:, start:end] - input_data[:,self.norm_point[index]][:,None,:]
                    )
                index += 1
        return torch.cat(
            [input_data, origin_input_data[:, :, 2:6], conf.unsqueeze(-1)], dim=-1
        )

    def _build_augmentation_list(self):
        transforms = []
        for entry in self.augmentation_types:
            if isinstance(entry, str):
                name, override_args = entry, {}
            else:
                name = entry["type"]
                override_args = entry.get("args", {})
            if name not in self.AUGMENTATION_REGISTRY:
                raise ValueError(
                    f"augmentation_types '{name}' tidak dikenal. "
                    f"Pilihan: {list(self.AUGMENTATION_REGISTRY.keys())}"
                )
            cls = self.AUGMENTATION_REGISTRY[name]
            transforms.append(cls(**override_args))
        return transforms

    def pose_transform(self):
        downsample_tf = skeleton_augmentation.Downsample(
            ratio=self.downsampling_ratio,
            random_offset=(self.transform_mode == "train"),
        )

        if self.transform_mode == "train":
            transforms = []
            if self.downsampling and self.downsampling_position == "before":
                transforms.append(downsample_tf)            # O-group: downsample -> augment
            transforms.extend(self._build_augmentation_list())
            if self.downsampling and self.downsampling_position == "after":
                transforms.append(downsample_tf)             # D-group: augment -> downsample
        else:
            transforms = []
            if self.downsampling:
                transforms.append(downsample_tf)

        transforms.append(skeleton_augmentation.ToTensor())
        print(f"[{self.transform_mode}] pipeline: {[t.__class__.__name__ for t in transforms]}")
        return skeleton_augmentation.Compose(transforms)

    def __len__(self):
        return len(self.inputs_list)

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        length = [len(vid) for vid in video]
        max_len = max(length)
        video_length = torch.LongTensor(
            [np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video]
        )
        left_pad = 6
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
        max_len = max_len + left_pad + right_pad
        padded_video = [
            torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1),
                ),
                dim=0,
            )
            for vid in video
        ]
        padded_video = torch.stack(padded_video)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return {
                'x': padded_video,
                'len_x': video_length,
                'label': padded_label,
                'label_lgt': label_length,
                'origin_info': info
            }