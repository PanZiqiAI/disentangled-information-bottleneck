
# Transforms
import numpy as np


class Flatten(object):
    """
    To vector.
    """
    def __call__(self, image):
        return image.view(-1, )


class To32x32(object):
    """
    From 28x28 -> 32x32.
    """
    def __call__(self, x):
        assert x.shape == (28, 28, 1)
        x = np.concatenate([np.zeros(shape=(28, 2, 1), dtype=np.float32), x,
                            np.zeros(shape=(28, 2, 1), dtype=np.float32)], axis=1)
        x = np.concatenate([np.zeros(shape=(2, 32, 1), dtype=np.float32), x,
                            np.zeros(shape=(2, 32, 1), dtype=np.float32)], axis=0)
        return x
