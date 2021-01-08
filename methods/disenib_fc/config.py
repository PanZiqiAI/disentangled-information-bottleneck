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

    def _add_root_args(self):
        super(ConfigTrain, self)._add_root_args()
        # Dataset & model
        self.parser.add_argument("--dataset",                       type=str,   default='mnist')

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
        # Modules
        if args_dict['dataset'] == 'mnist':
            self.parser.set(["input_dim", "num_classes"], [784, 10])
        self.parser.add_argument("--enc_hidden_dims",               type=str,   default="[1024,1024]")
        self.parser.add_argument("--dec_hidden_dims",               type=str,   default="[]")
        self.parser.add_argument("--rec_hidden_dims",               type=str,   default="[1024,1024]")
        self.parser.add_argument("--style_dim",                     type=int,   default=16)
        self.parser.add_argument("--class_dim",                     type=int,   default=16)
        self.parser.add_argument("--style_std",                     type=float, default=0.1)
        self.parser.add_argument("--class_std",                     type=float, default=1.0)
        self.parser.add_argument("--emb_radius",                    type=float, default=3.0)
        self.parser.add_argument("--n_samples",                     type=int,   default=10)
        # Optimization & Lambda
        self.parser.add_argument("--n_times_main",                  type=int,   default=10)
        self.parser.add_argument("--n_times_est",                   type=int,   default=3)
        self.parser.add_argument("--est_thr",                       type=int,   default=3000)
        self.parser.add_argument("--est_batch_size",                type=int,   default=64)
        self.parser.add_argument("--est_style_std",                 type=float, default=0.1)
        self.parser.add_argument("--est_class_std",                 type=float, default=0.1)
        self.parser.add_argument("--est_style_optimize",            type=int,   default=4)
        self.parser.add_argument("--lambda_dec",                    type=float, default=1.0)
        self.parser.add_argument("--lambda_rec",                    type=float, default=10.0)
        self.parser.add_argument("--lambda_est",                    type=float, default=0.5)
        self.parser.add_argument("--lambda_est_zc",                 type=float, default=0.05)
        self.parser.add_argument("--lambda_wall",                   type=float, default=10.0)
        # Evaluating args
        self.parser.add_argument("--freq_step_eval_quant",          type=int,   default=5000)
        self.parser.add_argument("--eval_batch_size",               type=int,   default=2560)
        self.parser.add_argument("--eval_attack_epsilons",          type=str,   default='[0.1,0.2,0.3]')
        self.parser.add_argument("--eval_odin_out_data",            type=str,   default='gauss')
        self.parser.add_argument("--eval_odin_temper",              type=float, default=1000)
        self.parser.add_argument("--eval_odin_noise_mag",           type=float, default=0.0014)
        self.parser.add_argument("--eval_odin_num_delta",           type=int,   default=10000)

    def _add_additional_args(self):
        # Epochs & batch size
        self.parser.add_argument("--steps",                         type=int,   default=20000)
        self.parser.add_argument("--batch_size",                    type=int,   default=64)
        # Learning rate
        self.parser.add_argument("--learning_rate",                 type=float, default=0.0001)
        # Frequency
        self.parser.add_argument("--freq_iter_log",                 type=int,   default=4096)
        self.parser.add_argument("--freq_step_chkpt",               type=int,   default=1000)
