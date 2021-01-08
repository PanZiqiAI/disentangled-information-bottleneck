
import numpy as np
from torch.utils.data import Dataset, DataLoader
from shared_libs.dataset_apis.disentangling.datasets import Shapes
from shared_libs.custom_packages.custom_pytorch.operations import DataCycle


class ShapesWithLabel(Dataset):
    """
    Shapes dataset with instance being classification label.
    """
    def __init__(self):
        self._dataset = Shapes().subset_with_structured_factors({'scale': [5], 'orientation': [3, 36]}, supervised=True)

    def __len__(self):
        return self._dataset.__len__()

    def __getitem__(self, index):
        # Get data & label.
        data, label = self._dataset[index]
        # Get shape label.
        label = label[0].astype(np.int64)
        # Return
        return data, label


def generate_data(cfg):
    """
    Generate data.
    """
    def _get_dataloader(_dataset, **kwargs):
        return DataLoader(
            dataset=_dataset,
            batch_size=cfg.args.batch_size if 'batch_size' not in kwargs.keys() else kwargs['batch_size'],
            drop_last=cfg.args.dataset_drop_last, shuffle=cfg.args.dataset_shuffle, num_workers=cfg.args.dataset_num_threads)

    # Return
    return {
        'train_data': DataCycle(_get_dataloader(ShapesWithLabel())),
        'train_est_data': DataCycle(_get_dataloader(ShapesWithLabel(), batch_size=cfg.args.est_batch_size))
    }
