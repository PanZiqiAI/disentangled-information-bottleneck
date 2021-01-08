# Configuration

import os
from shared_libs.custom_packages.custom_io.config import CanonicalConfigTrainPyTorch


########################################################################################################################
# Config for Train
########################################################################################################################

class ConfigTrain(CanonicalConfigTrainPyTorch):
    """
    The config for training models.
    """
    def __init__(self):
        super(ConfigTrain, self).__init__(
            os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../STORAGE/experiments'))

    def _init_method(self):
        # Check compatibility
        assert ',' not in self.args.gpu_ids, "Multi-GPU training is not developed. "

    def _set_directory_args(self, **kwargs):
        dirs = super(ConfigTrain, self)._set_directory_args()
        self.args.eval_dis_dir = os.path.join(self.args.ana_dir, 'disentangling')
        return dirs + [self.args.eval_dis_dir]

    def _add_root_args(self):
        super(ConfigTrain, self)._add_root_args()
        # Dataset
        self.parser.set(["dataset", "num_classes"], ["shapes", 3])

    def _add_tree_args(self, args_dict):
        ################################################################################################################
        # Datasets
        ################################################################################################################
        self.parser.add_argument("--dataset_shuffle",               type=int,   default=1,  choices=[0, 1])
        self.parser.add_argument("--dataset_num_threads",           type=int,   default=0)
        self.parser.add_argument("--dataset_drop_last",             type=bool,  default=True)
        ################################################################################################################
        # Others
        ################################################################################################################
        self.parser.add_argument("--style_dim",                     type=int,   default=16)
        self.parser.add_argument("--class_dim",                     type=int,   default=16)
        self.parser.add_argument("--style_std",                     type=float, default=0.1)
        self.parser.add_argument("--class_std",                     type=float, default=1.0)
        self.parser.add_argument("--emb_radius",                    type=float, default=3.0)
        # Optimization & Lambda
        self.parser.add_argument("--n_times_main",                  type=int,   default=5)
        self.parser.add_argument("--n_times_est",                   type=int,   default=3)
        self.parser.add_argument("--est_thr",                       type=int,   default=3000)
        self.parser.add_argument("--est_batch_size",                type=int,   default=64)
        self.parser.add_argument("--est_style_std",                 type=float, default=0.1)
        self.parser.add_argument("--est_class_std",                 type=float, default=0.1)
        self.parser.add_argument("--est_style_optimize",            type=int,   default=4)
        self.parser.add_argument("--lambda_dec",                    type=float, default=1.0)
        self.parser.add_argument("--lambda_rec",                    type=float, default=100.0)
        self.parser.add_argument("--lambda_est",                    type=float, default=1.0)
        self.parser.add_argument("--lambda_est_zc",                 type=float, default=0.05)
        self.parser.add_argument("--lambda_wall",                   type=float, default=10.0)
        # Evaluating args
        self.parser.add_argument("--freq_step_eval",                type=int,   default=500)
        self.parser.add_argument("--eval_dis_n_samples",            type=int,   default=10)

    def _add_additional_args(self):
        # Epochs & batch size
        self.parser.add_argument("--steps",                         type=int,   default=20000)
        self.parser.add_argument("--batch_size",                    type=int,   default=64)
        # Learning rate
        self.parser.add_argument("--learning_rate",                 type=float, default=0.0001)
        # Frequency
        self.parser.add_argument("--freq_iter_log",                 type=int,   default=4096)
        self.parser.add_argument("--freq_step_chkpt",               type=int,   default=1000)
