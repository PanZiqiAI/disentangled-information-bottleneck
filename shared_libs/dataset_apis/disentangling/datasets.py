
import os
import numpy as np
from skimage.io import imread
from shared_libs.dataset_apis import __DATA_ROOT__


########################################################################################################################
# Abstract Dataset
########################################################################################################################

class Datasets(object):
    """
    Super class for datasets.
    """

    def __init__(self, name, factors_info, dataset, supervised=False):
        """
        :param name:
        :param factors_info: Tuple of (factors_structure, factors_name):
            - factors_structure: [num_factor1_values, num_factor2_values, ..., num_factorN_values]
            - factor_name: [factor1_name, factor2_name, ..., factorN_name]
        :param dataset: Will be np.array in the shape of (num_data, ...)
            - Str: Load from file.
            - np.array: Directly given.
        :param supervised: Whether to generate from factors_structure or not.
        """
        ################################################################################################################
        # 1. Dataset information
        ################################################################################################################
        # (1) Name
        self._name = name
        # (2) Factor info
        factors_structure, factors_name = factors_info
        # 1> Factors structure
        assert factors_structure is not None
        for num_factor_values in factors_structure:
            assert num_factor_values != 1, "A factor with single possible value is inadequate to be a factor. "
        self._factors_structure = factors_structure
        # 2> Factors name
        assert factors_name is not None and len(factors_name) == len(set(factors_name))
        self._factors_name = factors_name
        ################################################################################################################
        # 2. Dataset & label
        ################################################################################################################
        # (1) Load original dataset
        if isinstance(dataset, str):
            self._np_dataset = self._load_dataset_and_preproc(dataset_path=dataset)
        else:
            self._np_dataset = dataset
        # (2) Generate label
        if supervised:
            self._label = self._generate_category_label()
        else:
            self._label = None

    def subset_with_structured_factors(self, factor_info, supervised=False):
        """
        :param factor_info: { factor_name: [factor_value1, factor_value2, ..., ] }
        :param supervised:
        :return:
        """
        ################################################################################################################
        # 1. Generate sub dataset.
        ################################################################################################################
        # (1) Reshape to structure
        np_dataset_structured = np.reshape(
            self._np_dataset, newshape=(self._factors_structure + self._np_dataset.shape[1:]))
        # (2) Get result: processing each factor.
        for factor_name, factor_values in factor_info.items():
            factor_index = self._factors_name.index(factor_name)
            # Get current result
            np_dataset_cur_factor = None
            for factor_v in factor_values:
                # Get subset
                np_dataset_subset = eval('np_dataset_structured[%s%s%s]' % (
                    ':,' * factor_index,
                    '%d:%d' % (factor_v, factor_v + 1),
                    ',:' * (len(np_dataset_structured.shape) - 1 - factor_index)))
                # Save to result
                if np_dataset_cur_factor is None:
                    np_dataset_cur_factor = np_dataset_subset
                else:
                    np_dataset_cur_factor = np.concatenate((np_dataset_cur_factor, np_dataset_subset), axis=factor_index)
            # Update
            np_dataset_structured = np_dataset_cur_factor
        # (3) Flatten
        np_dataset = np.reshape(np_dataset_structured, newshape=(-1,) + self._np_dataset.shape[1:])
        ################################################################################################################
        # 2. Generator sub factors_info.
        ################################################################################################################
        # (1) Init result
        factors_structure, factors_name = [], []
        # (2) Collect. Process each factor
        for factor_index in range(len(self._factors_structure)):
            cur_factor_name = self._factors_name[factor_index]
            if cur_factor_name in factor_info.keys():
                # Get num_factor_values for current factor
                num_cur_factor_values = len(factor_info[cur_factor_name])
                # Save
                if num_cur_factor_values > 1:
                    factors_structure.append(num_cur_factor_values)
                    factors_name.append(self._factors_name[factor_index])
            else:
                factors_structure.append(self._factors_structure[factor_index])
                factors_name.append(cur_factor_name)
        # (3) Get result
        factors_structure = tuple(factors_structure)
        factors_name = tuple(factors_name)
        # Get subset.
        subset = self.__class__(
            name='%s_subset' % self._name, factors_info=(factors_structure, factors_name),
            dataset=np_dataset, supervised=supervised)
        # Return
        return subset

    def __len__(self):
        return len(self._np_dataset)

    ####################################################################################################################
    # Load dataset & label
    ####################################################################################################################

    def _load_dataset_and_preproc(self, dataset_path):
        raise NotImplementedError

    def _generate_category_label(self):
        """
            Given self._np_dataset in shape of (num_data, ...) with num_data=prod_j num_factor_j_values, label will
        be generated as (num_data, num_factors) with label[i, j] denotes the unit category label of j-th factor of
        i-th sample.
        :return: (num_data, num_factors)
        """
        # 1. Init result. (num_factor1_values, num_factor2_values, ..., num_factors)
        label = None
        # 2. Generate (num_values_factor1, ..., num_factor_j_values, ..., 1)
        for factor_index, num_factor_values in enumerate(self._factors_structure):
            # (1) Generate a range of (num_factor_values, )
            factor_label = np.arange(start=0, stop=num_factor_values, step=1, dtype=np.uint8)
            # (2) Expand to aforementioned format
            for j in range(len(self._factors_structure)):
                if j < factor_index: factor_label = np.expand_dims(factor_label, axis=0)
                if j > factor_index: factor_label = np.expand_dims(factor_label, axis=-1)
            factor_label = np.broadcast_to(factor_label, shape=self._factors_structure)
            factor_label = np.expand_dims(factor_label, axis=-1)
            # (3) Save to result
            if label is None:
                label = factor_label
            else:
                label = np.concatenate((label, factor_label), axis=-1)
        # 3. Reshape of (num_data, num_factors)
        label = np.reshape(label, newshape=(-1, label.shape[-1]))
        # Return
        return label

    ####################################################################################################################
    # Get batch data
    ####################################################################################################################

    def _data_preprocess(self, data):
        raise NotImplementedError

    def __getitem__(self, index):
        # 1. Get data & label
        if index == 'all':
            data = self._np_dataset
            label = self._label
        else:
            data = self._np_dataset[index]
            label = self._label[index] if self._label is not None else None
        # 2. Preprocess
        data = self._data_preprocess(data)
        # Return
        return data if label is None else (data, label)


########################################################################################################################
# Dataset Instances
########################################################################################################################

class Shapes(Datasets):
    """
    Shapes dataset.
    """
    def __init__(self, name='shapes', factors_info=((3, 6, 40, 32, 32), ('shape', 'scale', 'orientation', 'pos_x', 'pos_y')),
                 dataset=os.path.join(__DATA_ROOT__, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), supervised=False):
        super(Shapes, self).__init__(name=name, factors_info=factors_info, dataset=dataset, supervised=supervised)

    def _load_dataset_and_preproc(self, dataset_path):
        dataset = np.load(dataset_path, encoding='latin1')['imgs'][:, np.newaxis]
        return dataset

    def _data_preprocess(self, data):
        return data.astype('float32')


class Sprites(Datasets):
    """
    Sprites dataset.
    """
    def __init__(self, name='sprites', factors_info=((8, 4, 73), ('instance', 'viewpoint', 'frame')),
                 dataset=os.path.join(__DATA_ROOT__, 'sprites'), supervised=False):
        super(Sprites, self).__init__(name=name, factors_info=factors_info, dataset=dataset, supervised=supervised)

    def _load_dataset_and_preproc(self, dataset_path):

        def _split_images(_img):
            _img_size = 64
            # Get each animation. (viewpoints, n_frames)
            _spellcast = _img[:4*_img_size, :7*_img_size]
            _thrust = _img[4*_img_size:8*_img_size, :8*_img_size]
            _walk = _img[8*_img_size:12*_img_size, :9*_img_size]
            _slash = _img[12*_img_size:16*_img_size, :6*_img_size]
            _shoot = _img[16*_img_size:20*_img_size]
            # Return
            return [_spellcast, _thrust, _walk, _slash, _shoot]

        ret = []
        for ins_index in range(8):
            # Bow
            img_bow = imread(os.path.join(dataset_path, 'instance[%d]-bow.png' % ins_index))
            spellcast, thrust, walk, slash, shoot_bow = _split_images(img_bow)
            # Spear
            img_spear = imread(os.path.join(dataset_path, 'instance[%d]-spear.png' % ins_index))
            _, thrust_spear, walk_spear, slash, shoot = _split_images(img_spear)
            # Get instance result. (viewpoints*64, all_frames*64, 4), where all_frames=7+8*2+9*2+6+13*2=73
            ret_ins = np.concatenate(
                [spellcast, thrust, thrust_spear, walk, walk_spear, slash, shoot, shoot_bow], axis=1)
            # ----------------------------------------------------------------------------------------------------------
            # 1. Split to (viewpoints, 64, all_frames*64, 4)
            ret_ins_views = np.array_split(ret_ins, 4, axis=0)
            ret_ins = np.concatenate([ret_ins_v[np.newaxis, ] for ret_ins_v in ret_ins_views], axis=0)
            # 2. Split to (viewpoints, 64, 64, 4) * all_frames -> (viewpoints, all_frames, 64, 64, 4)
            ret_ins_frames = np.array_split(ret_ins, 73, axis=2)
            ret_ins = np.concatenate([ret_ins_f[:, np.newaxis, ] for ret_ins_f in ret_ins_frames], axis=1)
            # Flatten. (n, 64, 64, 4)
            ret_ins = np.reshape(ret_ins, newshape=(ret_ins.shape[0] * ret_ins.shape[1], ) + ret_ins.shape[2:])
            # ----------------------------------------------------------------------------------------------------------
            # To (n, 60, 60, 3) -> (n, 3, 60, 60)
            ret_ins = (ret_ins[:, 8:, 2:62, :3] / 255.0).astype(np.float32)
            ret_ins = np.concatenate([ret_ins, np.zeros(shape=(ret_ins.shape[0], 4, 60, 3), dtype=np.float32)], axis=1)
            ret_ins = ret_ins.swapaxes(2, 3).swapaxes(1, 2)
            # Save
            ret.append(ret_ins)
        ret = np.concatenate(ret, axis=0)
        # Return
        return ret

    def _data_preprocess(self, data):
        return data
