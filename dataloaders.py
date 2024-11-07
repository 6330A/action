import os
from torch.utils.data import Dataset
import pickle
import numpy as np
from torchvision import transforms
from custom_transforms import *
from utils import *


class Le2i(Dataset):
    def __init__(self, sample_list, pose_encoding_path, label_path, opts, transform_type):
        self.sample_list = sample_list
        self.pose_encoding_path = pose_encoding_path
        self.label_path = label_path
        self.opts = opts
        self.transform_type = transform_type

        """sample_list代表的样本的下标，1-1808,对应04d的npy文件，也对应label"""
        use_flip = opts.use_flip if transform_type == 'train' else 'False'
        use_paa = opts.paa if transform_type == 'train' else 'False'

        self.flip_potion_transform = PotionFlip(use_flip, 0.5, opts.pose_type)
        if opts.normalize_type == 'max':
            self.normalize_transform = Normalize_max(opts.normalize)
        elif opts.normalize_type == 'area':
            self.normalize_transform = Normalize_area(opts.normalize, opts.channels)
        self.normalize_display = transforms.Compose([Normalize_max(opts.normalize), transforms.ToTensor()])

        if opts.paa_type == 'joint_wise':
            self.paa_transform = JointWiseTranslation(use_paa, max_motion=opts.max_motion)
        elif opts.paa_type == 'global':
            self.paa_transform = GlobalTranslation(use_paa, max_motion=opts.max_motion)
        elif opts.paa_type == 'group_wise':
            self.paa_transform = GroupWiseTranslation(use_paa, max_motion=opts.max_motion)
        elif opts.paa_type == 'global_and_groupwise':
            self.paa_transform = transforms.Compose([GlobalTranslation(use_paa, max_motion=opts.max_motion),
                                                     GroupWiseTranslation(use_paa, max_motion=opts.max_motion_groupwise)])
        # if opts.pose_type == 'openpose_coco_v2':
        self.potion_path = self.pose_encoding_path

        self.channels = opts.channels
        self.use_flip = opts.use_flip

    def __len__(self):
        return len(self.sample_list)

    def class_labels(self):
        return ['notfall', 'fall']

    def joint_names(self):
        return ['Nose', 'REye', 'LEye', 'REar', 'LEar', 'RSh', 'LSh', 'RElb', 'LElb', 'RHand', 'LHand', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RFoot', 'LFoot', 'BKG', 'CNTR']

    def __getitem__(self, idx):
        realindex = self.sample_list[idx]  # 样本的下标,用于获取样本、标签具体下标和

        potion_path_for_video = os.path.join(self.potion_path, str(realindex + 1).zfill(4))
        trajectory = np.load(potion_path_for_video + '.npy')

        # 对于部分宽度为113的元素进行裁剪
        if trajectory.shape[3] == 113:
            i = (113 - 85) // 2
            j = i + 85
            trajectory = trajectory[:, :, :, i:j]

        labels = np.load(self.label_path)
        label = labels[realindex]

        # 根据整体进行一个归一化，一般形状为(C, J, H, W) = (3, 19, 64, 86)，其中C通道数RGB是帧序列转为颜色而来，矩阵相乘matrix[3, frames] * matrix[frames, pix]
        # 归一化normalize_transform的参数frames需要匹配，对于跳跃取帧，需要处理，要求与npy文件生成时候的帧数匹配
        after_norm = self.normalize_transform(trajectory, frames=40)  # Normalize_area实例

        after_motion = self.paa_transform(after_norm)  # 随机抖动，就是随机平移一下，一个是以关节组为单位抖动，一个是全体统一抖动
        after_transform = self.flip_potion_transform(after_motion)  # 随机翻转，关节维度交换位置即可，类似镜像，更像是人转身而已
        if self.opts.return_augmented_view == 'True':
            after_motion2 = self.paa_transform(after_norm)
        after_transform2 = self.flip_potion_transform(after_motion2)

        sample = {'label': label,
                  'idx': idx,
                  'motion_rep': after_transform,
                  'motion_rep_augmented': after_transform2}
        return sample


class JHMDB(Dataset):
    def __init__(self, data_loc, pose_encoding_path, file_name, opts, transform_type):

        self.file_name = file_name
        self.data = pickle.load(open(data_loc + file_name + '.pkl', 'rb'))
        self.pose_encoding_path = pose_encoding_path
        if transform_type == 'train':
            use_flip = opts.use_flip
            use_paa = opts.paa
        else:
            use_flip = 'False'
            use_paa = 'False'
        self.flip_potion_transform = PotionFlip(use_flip, 0.5, opts.pose_type)
        if opts.normalize_type == 'max':
            self.normalize_transform = Normalize_max(opts.normalize)
        elif opts.normalize_type == 'area':
            self.normalize_transform = Normalize_area(opts.normalize, opts.channels)
        self.normalize_display = transforms.Compose([Normalize_max(opts.normalize), transforms.ToTensor()])

        if opts.paa_type == 'joint_wise':
            self.paa_transform = JointWiseTranslation(use_paa, max_motion=opts.max_motion)
        elif opts.paa_type == 'global':
            self.paa_transform = GlobalTranslation(use_paa, max_motion=opts.max_motion)
        elif opts.paa_type == 'group_wise':
            self.paa_transform = GroupWiseTranslation(use_paa, max_motion=opts.max_motion)
        elif opts.paa_type == 'global_and_groupwise':
            self.paa_transform = transforms.Compose([GlobalTranslation(use_paa, max_motion=opts.max_motion), \
                                                     GroupWiseTranslation(use_paa, max_motion=opts.max_motion_groupwise)])
        if opts.pose_type == 'openpose_coco_v2':
            self.potion_path = f'{self.pose_encoding_path}/openpose_COCO_' + str(opts.channels)

        self.channels = opts.channels
        self.opts = opts
        self.transform_type = transform_type
        self.use_flip = opts.use_flip

    def __len__(self):
        return len(self.data['labels'])

    def class_labels(self):
        return self.data['class_labels']

    def joint_names(self):
        return ['Nose', 'REye', 'LEye', 'REar', 'LEar', 'RSh', 'LSh', 'RElb', 'LElb', 'RHand', 'LHand', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RFoot', 'LFoot', 'BKG', 'CNTR']

    def __getitem__(self, idx):
        no_of_frames = len(self.data['frames'][idx])
        potion_path_for_video = os.path.join(self.potion_path, self.data['video_name'][idx])
        trajectory = np.load(potion_path_for_video + '.npy')
        after_norm = self.normalize_transform(trajectory, frames=no_of_frames)
        after_motion = self.paa_transform(after_norm)
        after_transform = self.flip_potion_transform(after_motion)
        if self.opts.return_augmented_view == 'True':
            after_motion2 = self.paa_transform(after_norm)
            after_transform2 = self.flip_potion_transform(after_motion2)

        label = self.data['labels'][idx]
        sample = {'label': label,
                  'video_name_actual': self.data['video_name'][idx],
                  'idx': idx,
                  'motion_rep': after_transform,
                  'motion_rep_augmented': after_transform2}
        return sample


class HMDB(Dataset):
    def __init__(self, data_loc, pose_encoding_path, file_name, opts, transform_type, get_whole_video=False):

        self.file_name = file_name
        self.data = pickle.load(open(data_loc + file_name + '.pkl', 'rb'))
        self.pose_encoding_path = pose_encoding_path

        if transform_type == 'train':
            use_flip = opts.use_flip
            use_paa = opts.paa
        else:
            use_flip = 'False'
            use_paa = 'False'

        self.random_crop = RandomCrop((64, 86))
        self.center_crop = CenterCrop((64, 86))

        self.flip_potion_transform = PotionFlip(use_flip, 0.5, opts.pose_type)
        if opts.normalize_type == 'max':
            self.normalize_transform = Normalize_max(opts.normalize)
        elif opts.normalize_type == 'area':
            self.normalize_transform = Normalize_area(opts.normalize, opts.channels)

        # self.paa_transform = MovePotion(use_paa=opts.paa if transform_type == 'train' else 'False',max_motion=opts.max_motion)
        if opts.paa_type == 'joint_wise':
            self.paa_transform = JointWiseTranslation(use_paa, max_motion=opts.max_motion)
        elif opts.paa_type == 'global':
            self.paa_transform = GlobalTranslation(use_paa, max_motion=opts.max_motion)
        elif opts.paa_type == 'group_wise':
            self.paa_transform = GroupWiseTranslation(use_paa, max_motion=opts.max_motion)
        elif opts.paa_type == 'global_and_groupwise':
            self.paa_transform = transforms.Compose([GlobalTranslation(use_paa, max_motion=opts.max_motion), \
                                                     GroupWiseTranslation(use_paa, max_motion=opts.max_motion_groupwise)])

        if opts.pose_type == 'openpose_coco_v2':
            self.potion_path = f'{self.pose_encoding_path}/openpose_COCO_' + str(opts.channels)

        self.channels = opts.channels
        self.opts = opts
        self.transform_type = transform_type
        self.use_flip = opts.use_flip
        self.get_whole_video = get_whole_video

    def __len__(self):
        return len(self.data['labels'])

    def class_labels(self):
        return self.data['class_labels']

    def joint_names(self):
        return ['Nose', 'REye', 'LEye', 'REar', 'LEar', 'RSh', 'LSh', 'RElb', 'LElb', 'RHand', 'LHand', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RFoot', 'LFoot', 'BKG', 'CNTR']

    def __getitem__(self, idx):
        no_of_frames = len(self.data['frames'][idx])
        potion_path_for_video = os.path.join(self.potion_path, self.data['video_name'][idx])
        trajectory = np.load(potion_path_for_video + '.npy')  # Frames x 17 x 64 x 86
        if self.transform_type == 'train':
            trajectory = self.random_crop(trajectory)
        elif self.transform_type == 'val' and not self.get_whole_video:
            trajectory = self.center_crop(trajectory)
        after_norm = self.normalize_transform(trajectory, frames=no_of_frames)
        after_motion = self.paa_transform(after_norm)
        after_transform = self.flip_potion_transform(after_motion)
        if self.opts.return_augmented_view == 'True':
            after_motion2 = self.paa_transform(after_norm)
            after_transform2 = self.flip_potion_transform(after_motion2)

        label = self.data['labels'][idx]
        sample = {'label': label,
                  'video_name_actual': self.data['video_name'][idx],
                  'idx': idx,
                  'motion_rep': after_transform,
                  'motion_rep_augmented': after_transform2}

        return sample
