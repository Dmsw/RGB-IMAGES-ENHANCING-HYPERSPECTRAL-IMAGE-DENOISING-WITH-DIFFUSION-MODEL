import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from os.path import join
import os


def load_icvl_data(
        *,
        resolution,
        data_dir,
        batch_size,
        deterministic=False,
):
    image_paths = []
    for f in os.listdir(data_dir):
        ext = f.split(".")[-1]
        if ext == "npz":
            image_paths.append(join(data_dir, f))

    dataset = ICVLDataset(
        resolution=resolution,
        image_paths=image_paths,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_hsi_data(
        *,
        data_dir,
        batch_size,
        deterministic=False,
        mode="train",
):
    dataset = HSIDataset(
        mode=mode,
        data_path=data_dir,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_flip=(mode == "train"),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_rgb_data(
        *,
        data_dir,
        batch_size,
        deterministic=False,
        mode="train",
):
    dataset = RGBDataset(
        mode=mode,
        data_path=data_dir,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_flip=True,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_data(
        *,
        data_dir,
        batch_size,
        image_size,
        class_cond=False,
        deterministic=False,
        random_crop=False,
        random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ICVLDataset(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            shard=0,
            num_shards=1,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        data = np.load(path)
        # clean_img = data["clean_img"][:1344, :1280]
        # noise_img = data["noise_img"][:1344, :1280]
        clean_img, noise_img = self._random_crop(data["clean_img"], data["noise_img"])
        noise_patch = noise_img.astype(np.float32) * 2 - 1
        clean_patch = clean_img.astype(np.float32) * 2 - 1
        out_dict = {}
        return [np.transpose(clean_patch, [2, 0, 1]), np.transpose(noise_patch, [2, 0, 1])], out_dict
    
    def _random_crop(self, img1, img2):
        imgw, imgh = img1.shape[:2]
        start_h = random.randint(0, imgh - self.resolution)
        start_w = random.randint(0, imgw - self.resolution)
        patch1 = img1[
                start_h:start_h + self.resolution,
                start_w:start_w + self.resolution
                ]
        patch2 = img2[
                start_h:start_h + self.resolution,
                start_w:start_w + self.resolution
                ]
        return patch1, patch2


class ImageDataset(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class RGBDataset(Dataset):
    def __init__(self,
                 resolution=256,
                 mode="train",
                 data_path="/home/root/dataset/cave/randga75/",
                 shard=0,
                 num_shards=1,
                 random_flip=False,
                 ):
        super().__init__()
        if mode == "train":
            self.data_path = join(data_path, "train.npz")
        else:
            self.data_path = join(data_path, "test.npz")
        data = np.load(self.data_path)
        self.mode = mode
        self.rgb_img = data["rgb_img"][shard:][::num_shards]
        self.resolution = resolution
        self.random_flip = random_flip
        self.n_scenario = self.rgb_img.shape[0]
        self.n_spectrum = self.rgb_img.shape[3]

    def __len__(self):
        return self.n_scenario

    def __getitem__(self, idx):
        if self.mode == "train":
            img = self.rgb_img[idx]
            arr = self._random_crop(img)
            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]
            if self.random_flip and random.random() < 0.5:
                arr = arr[::-1, :]
            arr = arr.astype(np.float32) * 2 - 1
            arr = np.transpose(arr, [2, 0, 1])
            mask = self._no_mask(arr)
            if mask is None:
                return {"input": arr}, {}
            else:
                return {"input":arr, "mask":mask}, {}
        else:
            noise_img = self.noise_img[idx]
            clean_img = self.clean_img[idx]
            noise_patch = noise_img.astype(np.float32) * 2 - 1
            clean_patch = clean_img.astype(np.float32) * 2 - 1
            out_dict = {}
            return [np.transpose(clean_patch, [2, 0, 1]), np.transpose(noise_patch, [2, 0, 1])], out_dict

    def _no_mask(self, arr):
        return None

    def _random_crop(self, img):
        imgw, imgh = img.shape[:2]
        start_h = random.randint(0, imgh - self.resolution)
        start_w = random.randint(0, imgw - self.resolution)
        patch = img[
                start_h:start_h + self.resolution,
                start_w:start_w + self.resolution
                ]
        return patch

    def _crop_img(self, img):
        imgw, imgh = img.shape[:2]
        self.crop_shape = (imgw // self.resolution, imgh // self.resolution)
        self.n_patch = self.crop_shape[0] * self.crop_shape[1]
        patch = np.zeros([self.n_patch, self.resolution, self.resolution, 1])
        for i in range(self.n_patch):
            ih = i // self.crop_shape[0]
            iw = i - ih * self.crop_shape[0]
            patch[i] = img[iw * self.resolution:iw * self.resolution + self.resolution,
                       ih * self.resolution:ih * self.resolution + self.resolution]
        return patch



class HSIDataset(Dataset):
    def __init__(self,
                 resolution=256,
                 mode="train",
                 data_path="/home/root/dataset/cave/randga25/",
                 shard=0,
                 num_shards=1,
                 random_flip=False,
                 ):
        super().__init__()
        if mode == "train":
            self.data_path = join(data_path, "train.npz")
        else:
            self.data_path = join(data_path, "test.npz")
        data = np.load(self.data_path)
        self.mode = mode
        self.noise_img = data["noise_img"][shard:][::num_shards]
        self.clean_img = data["clean_img"][shard:][::num_shards]
        self.resolution = resolution
        self.random_flip = random_flip
        self.n_scenario = self.noise_img.shape[0]
        self.n_spectrum = self.noise_img.shape[3]
        self.crop_shape = (1, 1)
        self.n_patch = 1
        self.mask_prob = 0.5

    def __len__(self):
        return self.n_scenario

    def __getitem__(self, idx):
        if self.mode == "train":
            # idx = idx // (512//self.resolution)**2
            img = self.clean_img[idx]
            arr = self._random_crop(img)
            # arr = img
            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]
            if self.random_flip and random.random() < 0.5:
                arr = arr[::-1, :]
            arr = arr.astype(np.float32) * 2 - 1
            arr = np.transpose(arr, [2, 0, 1])
            mask = self._no_mask(arr)
            if mask is None:
                return {"input": arr}, {}
            else:
                return {"input":arr, "mask":mask}, {}
        else:
            noise_img = self.noise_img[idx]
            clean_img = self.clean_img[idx]
            noise_patch = noise_img.astype(np.float32) * 2 - 1
            clean_patch = clean_img.astype(np.float32) * 2 - 1
            out_dict = {}
            return [np.transpose(clean_patch, [2, 0, 1]), np.transpose(noise_patch, [2, 0, 1])], out_dict

    def _no_mask(self, arr):
        return None

    def _random_crop(self, img):
        imgw, imgh = img.shape[:2]
        start_h = random.randint(0, imgh - self.resolution)
        start_w = random.randint(0, imgw - self.resolution)
        patch = img[
                start_h:start_h + self.resolution,
                start_w:start_w + self.resolution
                ]
        return patch

    def _crop_img(self, img):
        imgw, imgh = img.shape[:2]
        self.crop_shape = (imgw // self.resolution, imgh // self.resolution)
        self.n_patch = self.crop_shape[0] * self.crop_shape[1]
        patch = np.zeros([self.n_patch, self.resolution, self.resolution, 1])
        for i in range(self.n_patch):
            ih = i // self.crop_shape[0]
            iw = i - ih * self.crop_shape[0]
            patch[i] = img[iw * self.resolution:iw * self.resolution + self.resolution,
                       ih * self.resolution:ih * self.resolution + self.resolution]
        return patch



def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
