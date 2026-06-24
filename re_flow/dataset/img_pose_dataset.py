# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
from PIL import Image
import numpy as np


class Image_Pose_Dataset(torch.utils.data.Dataset):
    def __init__(self, information_path, root_path, keypoints_body_path, transform=None,
                 num_frames=16, joint='keypoint_vedio', joint_score='keypoint_score_vedio',
                 target_pose='joint', target_pose_score='joint_score', **kwargs):

        self.image_list, self.labels = self.make_img_path(information_path, root_path)

        self.keypoints_body_path = keypoints_body_path
        self.transform = transform
        self.num_frames = num_frames
        self.joint = joint
        self.joint_score = joint_score
        self.target_pose = target_pose
        self.target_pose_score = target_pose_score
        self.root_path = root_path

    def __len__(self):
        return len(self.image_list)

    def make_img_path(self, file_path1, input):
        # read txt
        img_path = []
        label = []

        with open(file_path1, 'r') as file:
            for line in file:
                # 移除行尾的换行符并分割字符串
                parts = line.strip().split()
                if parts:
                    # 添加第一列的数据
                    img_path.append(os.path.join(input, parts[0]))
                    # 添加最后一列的数据
                    label.append(int(parts[-1]))

        return img_path, label


    def _uniform_sample_indices(self, total_frames):
        skip = total_frames // self.num_frames
        if skip == 0:
            frame_id_list = np.arange(total_frames)
            res_id_list = np.ones(self.num_frames-total_frames) * (total_frames - 1)
            frame_id_list = np.concatenate([frame_id_list, res_id_list], axis=0)

        else:
            start_idx = int(np.random.randint(total_frames - (self.num_frames - 1) * skip, size=1))
            frame_id_list = np.arange(start_idx, start_idx + skip * self.num_frames, skip)

        return frame_id_list.astype('int32'), frame_id_list.astype('int32').copy()


    def _load_video(self, directory, frame_id_list):
        sampled_list = []
        if 'CSL' in directory:
            for _, path in enumerate(frame_id_list):
                img_path = os.path.join(directory, f'frame_{path:06d}.jpg')
                img = Image.open(img_path).convert('RGB')
                sampled_list.append(img)
        elif 'WLASL' in directory:
            for _, path in enumerate(frame_id_list):
                img_path = os.path.join(directory, f'{path+1:04d}.jpg')
                img = Image.open(img_path).convert('RGB')
                sampled_list.append(img)

        return sampled_list

    def _load_pose(self, directory, frame_id_list):
        video_name = directory.split('\\')
        # 1-75-86-81-32, 72-75, 78-75, 32-30-28, 36-34-32, 60-63, 66-69
        face_idx = [74, 85, 80, 31, 71, 77, 29, 27, 35, 33, 59, 62, 65, 68] # 14 nodes
        with np.load(os.path.join(self.keypoints_body_path, video_name[-1] + '.npz')) as data:
            keypoints = np.concatenate((data[self.joint][frame_id_list, :11, :],
                                        data[self.joint][frame_id_list, -42:, :],
                                        data[self.joint][frame_id_list][:, face_idx, :]),
                                       axis=1)
            keypoint_scores = np.concatenate((data[self.joint_score][frame_id_list, :11],
                                              data[self.joint_score][frame_id_list, -42:],
                                              data[self.joint_score][frame_id_list][:, face_idx]),
                                       axis=1)

        # with np.load(os.path.join(self.keypoints_body_path, video_name[-1] + '.npz')) as data:
        #     keypoints = np.concatenate((data[self.joint][frame_id_list, :11, :],
        #                                 data[self.joint][frame_id_list, -42:, :]),
        #                                axis=1)
        #     keypoint_scores = np.concatenate((data[self.joint_score][frame_id_list, :11],
        #                                       data[self.joint_score][frame_id_list, -42:]),
        #                                      axis=1)

        return torch.tensor(keypoints), torch.tensor(keypoint_scores)

    def __getitem__(self, idx):
        label = self.labels[idx]
        directory = self.image_list[idx]

        # 获取目录下的所有文件和目录名
        entries = os.listdir(directory)
        result = dict()
        # 计算文件数量
        file_count = sum(os.path.isfile(os.path.join(directory, entry)) for entry in entries)

        frame_id_list_1, frame_id_list_2 = self._uniform_sample_indices(file_count)
        images = self._load_video(directory, frame_id_list_1)
        keypoints, keypoint_scores = self._load_pose(directory, frame_id_list_2)

        if 'WLASL' in directory:
            ref_img_path = os.path.join(directory, '{:04d}.jpg'.format(torch.randint(1, file_count+1, size=())))
        elif 'CSL' in directory:
            ref_img_path = os.path.join(directory, 'frame_{:06d}.jpg'.format(torch.randint(0, file_count, size=())))
        ref_img = Image.open(ref_img_path).convert('RGB')
        keypoints = torch.flip(keypoints, dims=[-1])


        (result['image'], result[self.target_pose], result[self.target_pose_score], result['label'],
         result['reference_image']) = images, keypoints, keypoint_scores, label, ref_img
        del images, keypoints, keypoint_scores, label, ref_img

        result = self.transform(result)
        result['image'] = result['image'].view((self.num_frames, 3) + result['image'].size()[-2:])

        return result

