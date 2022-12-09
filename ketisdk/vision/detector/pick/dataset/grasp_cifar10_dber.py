from PIL import Image
import os
import os.path
import numpy as np
import sys
import torchvision.transforms as transforms

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url, check_integrity


class GraspCifar10(VisionDataset):

    url = None
    filename = None
    tgz_md5 = None
    train_list = [['data_batch', None]]

    test_list = [['test_batch', None]]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': None,
    }
    def __init__(self, root=None, train=True,
                 transform=None, target_transform=None,
                 download=False, im_shape=(32, 32, 3), data=None, indexing=False, base_folder='cifar10'):

        super(GraspCifar10, self).__init__(root)
        self.base_folder=base_folder
        self.transform = transform
        self.target_transform = target_transform

        self.indexing = indexing

        self.train = train  # training set or test set
        if download:
            self.download()

        if data is None:
            if not self._check_integrity():
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        if data is not None: self.targets = [0] * len(data)
        if data is None:
            # now load the picked numpy arrays
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, 'rb') as f:
                    if sys.version_info[0] == 2:
                        entry = pickle.load(f)
                    else:
                        # entry = pickle.load(f, encoding='latin1')
                        entry = pickle.load(f)
                    self.data.append(entry['data'])
                    if 'labels' in entry:
                        self.targets.extend(entry['labels'])
                    else:
                        self.targets.extend(entry['fine_labels'])

            self._load_meta()
            self.data = np.vstack(self.data).reshape(-1, im_shape[2], im_shape[0], im_shape[1])
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            self.data = np.vstack([p.reshape((1,) + p.shape) for p in data])


    def array2cifar(self, anArray):
        h, w, ch = anArray.shape
        out = [anArray[:, :, i].reshape(1, h * w) for i in range(ch)]
        return np.hstack(out)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.indexing:
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

