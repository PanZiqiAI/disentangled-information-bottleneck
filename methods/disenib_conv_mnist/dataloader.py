
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from shared_libs.dataset_apis.classification.datasets import ImageMNIST
from shared_libs.custom_packages.custom_pytorch.operations import DataCycle


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
        'train_data': DataCycle(_get_dataloader(
            ImageMNIST('train', cfg.args.dataset, transforms=ToTensor()))),
        'train_est_data': DataCycle(_get_dataloader(
            ImageMNIST('train', cfg.args.dataset, transforms=ToTensor()), batch_size=cfg.args.est_batch_size))
    }
