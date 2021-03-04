from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np


class CameraPoseDataset(Dataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, dataset_path, labels_file, data_transform=None, equalize_scenes=False):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :param data_transform: (Transform object) a torchvision transform object
        :return: an instance of the class
        """
        super(CameraPoseDataset, self).__init__()
        self.img_paths, self.poses, self.scenes, self.scenes_ids = read_labels_file(labels_file, dataset_path)
        self.dataset_size = self.poses.shape[0]
        self.num_scenes = np.max(self.scenes_ids) + 1
        self.scenes_sample_indices = [np.where(np.array(self.scenes_ids) == i)[0] for i in range(self.num_scenes)]
        self.scene_prob_selection = [len(self.scenes_sample_indices[i])/len(self.scenes_ids)
                                     for i in range(self.num_scenes)]
        if self.num_scenes > 1 and equalize_scenes:
            max_samples_in_scene = np.max([len(indices) for indices in self.scenes_sample_indices])
            unbalanced_dataset_size = self.dataset_size
            self.dataset_size = max_samples_in_scene*self.num_scenes
            num_added_positions = self.dataset_size - unbalanced_dataset_size
            # gap of each scene to maximum / # of added fake positions
            self.scene_prob_selection = [ (max_samples_in_scene-len(self.scenes_sample_indices[i]))/num_added_positions for i in range(self.num_scenes) ]
        self.transform = data_transform

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        if idx >= len(self.poses): # sample from an under-repsented scene
            sampled_scene_idx = np.random.choice(range(self.num_scenes), p=self.scene_prob_selection)
            idx = np.random.choice(self.scenes_sample_indices[sampled_scene_idx])

        img = imread(self.img_paths[idx])
        pose = self.poses[idx]
        scene = self.scenes_ids[idx]
        if self.transform:
            img = self.transform(img)

        sample = {'img': img, 'pose': pose, 'scene': scene}
        return sample


def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    scenes = df['scene'].values
    scene_unique_names = np.unique(scenes)
    scene_name_to_id = dict(zip(scene_unique_names, list(range(len(scene_unique_names)))))
    scenes_ids = [scene_name_to_id[s] for s in scenes]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return imgs_paths, poses, scenes, scenes_ids