from operator import mul
from functools import reduce

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

        if not check.is_tuple_of_ints(factor):
            raise ValueError("Factor must be an integer or tuple of integers.")

        if any([e<1 for e in factor]):
            raise ValueError("Factors must be >= 1.")

        self.factor = factor

    def get_output_shape(self, *shape):
        shape = list(shape)
        shape[0]  = shape[0] * reduce(mul, self.factor, 1)
        for i, f in enumerate(self.factor):
            i = -len(self.factor) + i
            shape[i] = shape[i] // f
        return tuple(shape)

    def forward(self, inputs, context=None):
        if inputs.dim() - len(self.factor) < 2:
            raise ValueError(f"Expecting inputs with {len(self.factor) + 2} dimensions, got {inputs.dim()}.")
        shape = inputs.size()
        output_shape = self.get_output_shape(*(shape[1:]))
        batch_size = shape[0]

        if any([(d%f != 0) for d,f in zip(shape[-len(self.factor):], self.factor)]):
            raise ValueError(f"Input image size {tuple(shape)} not compatible with the factor {self.factor}.")

        view = list(shape[:-len(self.factor)])
        for i, f in enumerate(self.factor):
            i = -len(self.factor) + i
            view += [shape[i] // f, f]
        inputs = inputs.view(*view)

        permutation = list(range(len(shape)+len(self.factor)))
        for i in range(len(self.factor)):
            permutation[2+i] = len(permutation)-2*len(self.factor)+2*i+1
            permutation[-len(self.factor)+i] = len(permutation)-2*len(self.factor)+2*i
        permutation[2+len(self.factor):-len(self.factor)] = list(range(len(self.factor),len(permutation)-2-len(self.factor)))

        inputs = inputs.permute(*permutation).contiguous()
        inputs = inputs.view(
            batch_size,
            *output_shape
        )

        return inputs, inputs.new_zeros(batch_size)

    def inverse(self, inputs, context=None):
        if inputs.dim() - len(self.factor) < 2:
            raise ValueError(f"Expecting inputs with {len(self.factor) + 2} dimensions")
        shape = inputs.size()
        batch_size = shape[0]
        c = shape[1]
        shape_out = list(shape)
        shape_out[1]  = c // reduce(mul, self.factor, 1)
        for i, f in enumerate(self.factor):
            i = -len(self.factor) + i
            shape_out[i] = shape[i] * f

        if c%reduce(mul, self.factor, 1) != 0:
            raise ValueError("Invalid number of channel dimensions.")

        inputs = inputs.view(
            batch_size, c // reduce(mul, self.factor, 1), *(self.factor + shape[2:])
        )
        permutation = list(range(len(shape)+len(self.factor)))
        for i in range(len(self.factor)):
            permutation[-2*len(self.factor)+2*i+1] = 2+i
            permutation[-2*len(self.factor)+2*i] = len(permutation)-len(self.factor)+i
        permutation[2:-2*len(self.factor)] = list(range(2+len(self.factor),len(permutation)-len(self.factor)))

        inputs = inputs.permute(*permutation).contiguous()
        inputs = inputs.view(*shape_out)

        return inputs, inputs.new_zeros(batch_size)
