
from torch.utils.data import DataLoader
from shared_libs.dataset_apis.classification.datasets import FlattenMNIST


def generate_data(cfg):
    """
    Generate data.
    """
    def _get_dataloader(_dataset, **kwargs):
        return DataLoader(
            dataset=_dataset,
            batch_size=cfg.args.batch_size if 'batch_size' not in kwargs.keys() else kwargs['batch_size'],
            drop_last=cfg.args.dataset_drop_last if 'drop_last' not in kwargs.keys() else kwargs['drop_last'],
            shuffle=cfg.args.dataset_shuffle, num_workers=cfg.args.dataset_num_threads)

    # Return
    return {
        'train_data': _get_dataloader(
            FlattenMNIST('train', cfg.args.dataset)),
        'eval_train_data': _get_dataloader(
            FlattenMNIST('train', cfg.args.dataset), batch_size=cfg.args.eval_batch_size, drop_last=False),
        'eval_test_data': _get_dataloader(
            FlattenMNIST('test', cfg.args.dataset), batch_size=cfg.args.eval_batch_size, drop_last=False)
    }
