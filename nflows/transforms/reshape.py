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

    def get_output_shape(self, *shape):
        shape = list(shape)
        shape[0]  = shape[0] * self.factor_x * self.factor_y
        shape[-2] = shape[-2] // self.factor_x
        shape[-1] = shape[-1] // self.factor_y
        return tuple(shape)

    def forward(self, inputs, context=None):
        if inputs.dim() < 4:
            raise ValueError("Expecting inputs with 4 dimensions")
        shape = inputs.size()
        output_shape = self.get_output_shape(*(shape[1:]))
        batch_size = shape[0]
        h = shape[-2]
        w = shape[-1]

        if h % self.factor_x != 0 or w % self.factor_y != 0:
            raise ValueError("Input image size not compatible with the factor.")

        inputs = inputs.view(
            *(shape[:-2]+(h // self.factor_x, self.factor_x, w // self.factor_y, self.factor_y))
        )
        permutation = list(range(len(shape)+2))
        permutation[2] = len(permutation)-3
        permutation[3] = len(permutation)-1
        permutation[-2] = len(permutation)-4
        permutation[-1] = len(permutation)-2
        permutation[4:-2] = list(range(2,len(permutation)-4))
        inputs = inputs.permute(*permutation).contiguous()
        inputs = inputs.view(
            batch_size,
            *output_shape
        )

        return inputs, inputs.new_zeros(batch_size)

    def inverse(self, inputs, context=None):
        if inputs.dim() < 4:
            raise ValueError("Expecting inputs with 4 dimensions")
        shape = inputs.size()
        batch_size = shape[0]
        c = shape[1]
        h = shape[-2]
        w = shape[-1]
        shape_out = list(shape)
        shape_out[1]  = c // self.factor_x // self.factor_y
        shape_out[-2] = h * self.factor_x
        shape_out[-1] = w * self.factor_y

        if c%(self.factor_x * self.factor_y) != 0:
            raise ValueError("Invalid number of channel dimensions.")

        inputs = inputs.view(
            batch_size, c // (self.factor_x * self.factor_y), self.factor_x, self.factor_y, *(shape[2:])
        )
        permutation = list(range(len(shape)+2))
        permutation[-4] = len(permutation)-2
        permutation[-3] = 2
        permutation[-2] = len(permutation)-1
        permutation[-1] = 3
        permutation[2:-4] = list(range(4,len(permutation)-2))
        inputs = inputs.permute(*permutation).contiguous()
        inputs = inputs.view(*shape_out)

        return inputs, inputs.new_zeros(batch_size)
