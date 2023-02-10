# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    """
    Apply a layer to every temporal slice of an input.
    Input should have at least 3 dimensions, with dimension 1 being temporal.
    """

    def __init__(self, module: nn.Module):
        """
        Params:
            module: torch module to apply to each timestep
        """

        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor):
        """
        Params:
            x: tensor ~ (batch_size, seq_len, ... )
        """
        assert len(x.shape) >= 3
        batch_size, seq_len = x.shape[:2]

        # flatten the timestep dimension to obtain (batch_size x seq_len, ..)
        x_reshape = x.contiguous().view(batch_size * seq_len, *x.shape[2:])

        # apply the module to the reshaped tensor
        y = self.module(x_reshape)

        # if y has two dimensions then it is ~ (batch_size*seq_len, out_dim)
        if len(y.shape) == 2:
            out_dim = y.shape[-1]
            output_shape = (1, out_dim)
        # otherwise, y has dimensions (batch_size*seq_len, c_out, out_dim)
        else:
            c_out, out_dim = y.shape[-2:]
            output_shape = (c_out, out_dim)

        # reshape the tensor to be (batch_size, seq_len, ...) again
        y = y.contiguous().view(batch_size, seq_len, *output_shape)

        return y
