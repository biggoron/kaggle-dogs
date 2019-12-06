from skimage.color import rgb2lab, lab2rgb
import torch

class RGB2LAB(object):
    """Converts RGB PIL image to LAB colorspace"""

    def __call__(self, sample):
        images = sample.numpy()

        # Create a new image by applying the transform object to the source image
        lab_image = rgb2lab(images.transpose(1, 2, 0))

        return torch.from_numpy(lab_image.transpose(2, 0, 1))

class LAB2RGB(object):
    """Converts LAB PIL image to RGB colorspace"""

    def __call__(self, sample):
        images = sample.numpy()

        # Create a new image by applying the transform object to the source image
        rgb_image = lab2rgb(images.transpose(1, 2, 0))

        return torch.from_numpy(rgb_image.transpose(2, 0, 1))
