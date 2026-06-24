import torch
import torchvision
import numpy as np
from PIL import Image
import random
from typing import Tuple, Dict, List, Optional, Sequence


class ImageScale(object):
    """Center cropping for images and Normalize the range of keypoint values"""
    def __init__(self, img_size: Tuple[int, int] = (256, 256),
                 target_size: int = 224,
                 image: str = 'image',
                 reference_image: str = 'reference_image') -> None:
        self.img_size = img_size
        self.target_size = target_size
        self.image = image
        self.reference_image = reference_image
        self.img_scale = torchvision.transforms.Resize(self.target_size, interpolation=Image.BICUBIC)
        self.img_crop = torchvision.transforms.CenterCrop(self.target_size)

    def __call__(self, results: Dict) -> Dict:
        pil_size = self.img_size
        while min(*pil_size) >= 2 * self.target_size:
            results[self.image] = [img.resize(tuple(x // 2 for x in pil_size), resample=Image.BOX)
                                   for img in results[self.image]]
            results[self.reference_image] = results[self.reference_image].resize(tuple(x // 2 for x in pil_size), resample=Image.BOX)
            pil_size = tuple(x // 2 for x in pil_size)

        results[self.image] = [self.img_scale(img) for img in results[self.image]]
        results[self.image] = [self.img_crop(img) for img in results[self.image]]
        results[self.reference_image] = self.img_scale(results[self.reference_image])
        results[self.reference_image] = self.img_crop(results[self.reference_image])

        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'img_shape={self.target_size})')
        return repr_str


class PoseNormalize(object):
    """Normalize the range of keypoint values.

    Required Keys:

        - keypoint
        - img_shape (optional)

    Modified Keys:

        - keypoint

    Args:
        img_shape (tuple[int, int]): The resolution of the original video.
            Defaults to ``(1080, 1920)``.
    """

    def __init__(self, img_shape: Tuple[int, int] = (256, 256),
                 joint: str = 'joint') -> None:
        self.img_shape = img_shape
        self.joint = joint

    def __call__(self, results: Dict) -> Dict:
        """The transform function of :class:`PreNormalize2D`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        h, w = self.img_shape
        results[self.joint][..., 0] = \
            (results[self.joint][..., 0] - (h / 2)) / (h / 2)
        results[self.joint][..., 1] = \
            (results[self.joint][..., 1] - (w / 2)) / (w / 2)
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'img_shape={self.img_shape})')
        return repr_str


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, probability: float = 0.5,
                 image: str = 'image',
                 joint: str = 'joint') -> None:
        self.probability = probability
        self.image = image
        self.joint = joint
        self.flip = torchvision.transforms.RandomHorizontalFlip(0.5)

    def __call__(self, results) -> Dict:
        if random.random() < self.probability:
            results[self.image] = [self.flip(img) for img in results[self.image]]
            results[self.joint] *= -1
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'probability={self.probability})')
        return repr_str


class JointToBone(object):
    """Convert the joint information to bone information.

    Required Keys:

        - keypoint

    Modified Keys:

        - keypoint

    Args:
        dataset (str): Define the type of dataset: 'nturgb+d', 'openpose',
            'coco'. Defaults to ``'nturgb+d'``.
        target (str): The target key for the bone information.
            Defaults to ``'keypoint'``.
    """

    def __init__(self, joint: str = 'joint',
                 joint_score: str = 'joint_score',
                 bone: str = 'bone',
                 bone_score: str = 'bone_score') -> None:
        self.joint = joint
        self.joint_score = joint_score
        self.bone = bone
        self.bone_score = bone_score
        self.pairs = torch.tensor([[0, 0], [1, 0], [2, 0], [3, 1], [5, 3], [7, 5], [9, 7],
                                   [4, 2], [6, 4], [8, 6], [10, 8],
                                   [11, 9], [12, 11], [13, 12], [14, 13], [15, 14],
                                   [16, 11], [17, 16], [18, 17], [19, 18],
                                   [20, 11], [21, 20], [22, 21], [23, 22],
                                   [24, 11], [25, 24], [26, 25], [27, 26],
                                   [28, 11], [29, 28], [30, 29], [31, 30],
                                   [32, 10], [33, 32], [34, 33], [35, 34], [36, 35],
                                   [37, 32], [38, 37], [39, 38], [40, 39],
                                   [41, 32], [42, 41], [43, 42], [44, 43],
                                   [45, 32], [46, 45], [47, 46], [48, 47],
                                   [49, 32], [50, 49], [51, 50], [52, 51]])

    def __call__(self, results: Dict) -> Dict:
        """The transform function of :class:`JointToBone`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        T, V, C = results[self.joint].shape
        assert C == 2
        bone = torch.zeros((T, V, C), dtype=torch.float32)
        bone_score = torch.zeros((T, V), dtype=torch.float32)

        for v1, v2 in self.pairs:
            bone[..., v1, :] = results[self.joint][..., v1, :] - results[self.joint][..., v2, :]
            bone_score = (results[self.joint_score][..., v1] + results[self.joint_score][..., v2]) / 2
        results[self.bone], results[self.bone_score] = bone, bone_score

        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'bone={self.bone})')
        return repr_str


class Stack(object):
    def __init__(self, roll: bool = True,
                 image: str = 'image',
                 joint: str = 'joint',
                 joint_score: str = 'joint_score',
                 bone: str = 'bone',
                 bone_score: str = 'bone_score',
                 reference_image: str = 'reference_image') -> None:
        self.roll = roll
        self.image = image
        self.joint = joint
        self.joint_score = joint_score
        self.bone = bone
        self.bone_score = bone_score
        self.reference_image = reference_image

    def __call__(self, results: Dict) -> Dict:
        results[self.image] = np.concatenate([np.array(x).transpose(2, 0, 1) for x in results[self.image]], axis=0)
        results[self.reference_image] = np.array(results[self.reference_image]).transpose(2, 0, 1)

        if self.bone in results.keys():
            results[self.joint] = torch.cat(
                (results[self.joint], results[self.joint_score][..., None],
                 results[self.bone], results[self.bone_score][..., None]), dim=-1)
            del results[self.joint_score], results[self.bone], results[self.bone_score]
        else:
            results[self.joint] = torch.cat(
                (results[self.joint], results[self.joint_score][..., None]), dim=-1)
            del results[self.joint_score]

        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'roll={self.roll})')
        return repr_str


class ToImageTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div: bool = True,
                 image: str = 'image',
                 reference_image: str = 'reference_image') -> None:
        self.div = div
        self.image = image
        self.reference_image = reference_image

    def __call__(self, results: Dict) -> Dict:
        results[self.image] = torch.from_numpy(results[self.image]).contiguous()
        results[self.image].float().div(255.) if self.div else results[self.image].float()
        results[self.reference_image] = torch.from_numpy(results[self.reference_image]).contiguous()
        results[self.reference_image].float().div(255.) if self.div else results[self.reference_image].float()

        return results


    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'div={self.div})')
        return repr_str


class TransformersToTensor(object):
    def __init__(self, image: str = 'image',
                 joint: str = 'joint',
                 joint_score: str = 'joint_score',
                 bone: str = 'bone',
                 bone_score: str = 'bone_score',
                 reference_image: str = 'reference_image') -> None:
        self.image = image
        self.reference_image = reference_image
        self.joint = joint
        self.joint_score = joint_score
        self.bone = bone
        self.bone_score = bone_score
        self.img_to_tensor = torchvision.transforms.ToTensor()


    def __call__(self, results: Dict) -> Dict:
        results[self.image] = torch.stack([self.img_to_tensor(img) for img in results[self.image]], dim=0)
        results[self.reference_image] = self.img_to_tensor(results[self.reference_image])


        if self.bone in results.keys():
            results[self.joint] = torch.cat(
                (results[self.joint], results[self.joint_score][..., None],
                 results[self.bone], results[self.bone_score][..., None]), dim=-1)
            del results[self.joint_score], results[self.bone], results[self.bone_score]
        else:
            results[self.joint] = torch.cat(
                (results[self.joint], results[self.joint_score][..., None]), dim=-1)
            del results[self.joint_score]

        return results

