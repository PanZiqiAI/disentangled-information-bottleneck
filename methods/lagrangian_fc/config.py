# Configuration

import os
from shared_libs.custom_packages.custom_basic.operations import chk_d
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
        self.parser.add_argument("--model",                         type=str,   default='vib',  choices=['vib', 'nib'])

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
        if args_dict['model'] == 'vib':
            self.parser.add_argument("--vib_softplus_scalar",       type=float, default=-1.0,
                                     help="Set to -1.0 to disable. To make the std initially small by minus a positive scalar.")
        if args_dict['model'] == 'nib':
            self.parser.add_argument("--nib_log_std",               type=float, default=1.0)
            self.parser.add_argument("--nib_log_std_trainable",     type=bool,  default=True)
        self.parser.add_argument("--enc_hidden_dims",               type=str,   default="[1024,1024]")
        self.parser.add_argument("--dec_hidden_dims",               type=str,   default="[]")
        self.parser.add_argument("--emb_dim",                       type=int,   default=16)
        # Optimization & Lambda
        self.parser.add_argument("--hfunc",                         type=str,   default='none', choices=['none', 'exp', 'pow'])
        if chk_d(args_dict, 'hfunc', '!=', 'none'):
            self.parser.add_argument("--hfunc_param",               type=float, default=1.0)
        self.parser.add_argument("--lambda_kl",                     type=float, default=0.01)
        # Evaluating args
        self.parser.add_argument("--eval_batch_size",               type=int,   default=2560)
        self.parser.add_argument("--eval_attack_epsilons",          type=str,   default='[0.1,0.2,0.3]')
        self.parser.add_argument("--eval_odin_out_data",            type=str,   default='gauss')
        self.parser.add_argument("--eval_odin_temper",              type=float, default=1000)
        self.parser.add_argument("--eval_odin_noise_mag",           type=float, default=0.0014)
        self.parser.add_argument("--eval_odin_num_delta",           type=int,   default=10000)

    def _add_additional_args(self):
        # Epochs & batch size
        self.parser.add_argument("--epochs",                        type=int,   default=100)
        self.parser.add_argument("--batch_size",                    type=int,   default=100)
        self.parser.add_argument("--early_stop_trials",             type=int,   default=10,
                                 help="Set to non-positive to disable. ")
        # Learning rate
        self.parser.add_argument("--learning_rate",                 type=float, default=0.0001)
        # Frequency
        self.parser.add_argument("--freq_iter_log",                 type=int,   default=4096)
