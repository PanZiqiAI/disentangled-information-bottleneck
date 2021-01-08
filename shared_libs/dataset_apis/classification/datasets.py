
import os
import numpy as np
import torchvision
from torch.utils.data import Dataset
from shared_libs.dataset_apis.classification import utils
from shared_libs.dataset_apis.classification import transforms as custom_transforms
from shared_libs.dataset_apis import __DATA_ROOT__


########################################################################################################################
# Dataset
########################################################################################################################

class BaseClassification(Dataset):
    """
    Base class for dataset for classification.
    """
    @property
    def classes(self):
        """
        :return: A list, whose i-th element is the name of the i-th category.
        """
        raise NotImplementedError

    @property
    def class_to_idx(self):
        """
        :return: A dict, where dict[key] is the category index corresponding to the category 'key'.
        """
        raise NotImplementedError

    @property
    def num_classes(self):
        """
        :return: A integer indicating number of categories.
        """
        raise NotImplementedError

    @property
    def class_counter(self):
        """
        :return: A list, whose i-th element equals to the total sample number of the i-th category.
        """
        raise NotImplementedError

    @property
    def sample_indices(self):
        """
        :return: A list, whose i-th element is a numpy.array containing sample indices of the i-th category.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        Should return x, y, where y is the class label.
        :param index:
        :return:
        """
        raise NotImplementedError


########################################################################################################################
# Instances
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# MNIST & FashionMNIST
# ----------------------------------------------------------------------------------------------------------------------

def mnist_paths(name):
    assert name in ['mnist', 'fashion_mnist']
    return {
        'train': (os.path.join(__DATA_ROOT__, "%s/train-images.idx3-ubyte" % name),
                  os.path.join(__DATA_ROOT__, "%s/train-labels.idx1-ubyte" % name)),
        'test': (os.path.join(__DATA_ROOT__, "%s/t10k-images.idx3-ubyte" % name),
                 os.path.join(__DATA_ROOT__, "%s/t10k-labels.idx1-ubyte" % name))}


class MNIST(BaseClassification):
    """
    MNIST dataset.
    """
    def __init__(self, images_path, labels_path, transforms=None):
        # Member variables
        self._transforms = transforms
        # Load from file
        # (1) Data & label
        self._dataset = utils.decode_idx3_ubyte(images_path).astype('float32')[:, :, :, np.newaxis] / 255.0
        self._label = utils.decode_idx1_ubyte(labels_path).astype('int64')
        # (2) Samples per category
        self._sample_indices = [np.argwhere(self._label == label)[:, 0].tolist() for label in range(self.num_classes)]
        # (3) Class counter
        self._class_counter = [len(samples) for samples in self._sample_indices]

    @property
    def num_classes(self):
        return len(set(self._label))

    @property
    def class_counter(self):
        return self._class_counter

    @property
    def sample_indices(self):
        return self._sample_indices

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        # 1. Get image & label
        image, label = self._dataset[index], self._label[index]
        # 2. Transform
        if self._transforms:
            image = self._transforms(image)
        # Return
        return image, label


class FlattenMNIST(MNIST):
    """
    Flatten MNIST dataset.
    """
    def __init__(self, phase, name='mnist', **kwargs):
        assert phase in ['train', 'test']
        # Get transforms
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, ), (0.5, )),
            custom_transforms.Flatten()]
        ) if 'transforms' not in kwargs.keys() else kwargs['transforms']
        # Init
        super(FlattenMNIST, self).__init__(*(mnist_paths(name)[phase]), transforms=transforms)


class ImageMNIST(MNIST):
    """
    Flatten MNIST dataset.
    """
    def __init__(self, phase, name='mnist', **kwargs):
        assert phase in ['train', 'test']
        # Get transforms
        if 'transforms' in kwargs.keys():
            transforms = kwargs['transforms']
        else:
            transforms = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, ), (0.5, ))]
            if 'to32x32' in kwargs.keys() and kwargs['to32x32']: transforms = [custom_transforms.To32x32()] + transforms
            transforms = torchvision.transforms.Compose(transforms)
        # Init
        super(ImageMNIST, self).__init__(*(mnist_paths(name)[phase]), transforms=transforms)
