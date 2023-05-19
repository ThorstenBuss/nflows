from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import RandomPermutation
from nflows.utils import torchutils


class OneByOneConvolution(LULinear):
    """An invertible 1x1 convolution with a fixed permutation, as introduced in the Glow paper.

    Reference:
    > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
    """

    def __init__(self, num_channels, using_cache=False, identity_init=True):
        super().__init__(num_channels, using_cache, identity_init)
        self.permutation = RandomPermutation(num_channels, dim=1)

    def _lu_forward_inverse(self, inputs, inverse=False):
        shape = inputs.shape[:1] + inputs.shape[2:] + inputs.shape[1:2]
        permutation = (0,) + tuple(range(2,inputs.dim())) + (1,)
        inverse_perm = (0,inputs.dim()-1) + tuple(range(1,inputs.dim()-1))
        inputs = inputs.permute(*permutation).reshape(-1, shape[-1])

        if inverse:
            outputs, logabsdet = super().inverse(inputs)
        else:
            outputs, logabsdet = super().forward(inputs)

        outputs = outputs.reshape(*shape).permute(*inverse_perm)
        logabsdet = logabsdet.reshape(*(shape[:-1]))

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        if inputs.dim() < 2:
            raise ValueError("Inputs must be a 4D tensor.")

        inputs, _ = self.permutation(inputs)

        return self._lu_forward_inverse(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        if inputs.dim() < 2:
            raise ValueError("Inputs must be a 4D tensor.")

        outputs, logabsdet = self._lu_forward_inverse(inputs, inverse=True)

        outputs, _ = self.permutation.inverse(outputs)

        return outputs, logabsdet
