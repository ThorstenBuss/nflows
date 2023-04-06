import torch

from nflows.transforms.base import Transform
import nflows.utils.typechecks as check


class SqueezeTransform(Transform):
    """A transformation defined for image data that trades spatial dimensions for channel
    dimensions, i.e. "squeezes" the inputs along the channel dimensions.

    Implementation adapted from https://github.com/pclucas14/pytorch-glow and
    https://github.com/chaiyujin/glow-pytorch.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, factor=2):
        super(SqueezeTransform, self).__init__()

        if check.is_int(factor):
            factor = (factor, factor)

        if check.is_list(factor):
            factor = tuple(factor)

        if not check.is_tuple_of_ints(factor) or len(factor)!=2:
            raise ValueError("Factor must be an integer or tuple of two integers.")

        if factor[0] < 1 or factor[1] < 1:
            raise ValueError("Factors must be >= 1.")

        self.factor_x, self.factor_y = factor

    def get_output_shape(self, c, h, w):
        return (c * self.factor_x * self.factor_y, h // self.factor_x, w // self.factor_y)

    def forward(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError("Expecting inputs with 4 dimensions")

        batch_size, c, h, w = inputs.size()

        if h % self.factor_x != 0 or w % self.factor_y != 0:
            raise ValueError("Input image size not compatible with the factor.")

        inputs = inputs.view(
            batch_size, c, h // self.factor_x, self.factor_x, w // self.factor_y, self.factor_y
        )
        inputs = inputs.permute(0, 1, 3, 5, 2, 4).contiguous()
        inputs = inputs.view(
            batch_size,
            c * self.factor_x * self.factor_y,
            h // self.factor_x,
            w // self.factor_y,
        )

        return inputs, inputs.new_zeros(batch_size)

    def inverse(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError("Expecting inputs with 4 dimensions")

        batch_size, c, h, w = inputs.size()

        if c < 4 or c % 4 != 0:
            raise ValueError("Invalid number of channel dimensions.")

        inputs = inputs.view(
            batch_size, c // (self.factor_x * self.factor_y), self.factor_x, self.factor_y, h, w
        )
        inputs = inputs.permute(0, 1, 4, 2, 5, 3).contiguous()
        inputs = inputs.view(
            batch_size, c // (self.factor_x * self.factor_y), h * self.factor_x, w * self.factor_y
        )

        return inputs, inputs.new_zeros(batch_size)
