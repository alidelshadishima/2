```python
# ============================================================
# operations_sr.py
#
# Search Space Operations for
# Single Image Super-Resolution (SISR)
#
# Dataset:
#     DIV2K
#
# Designed for:
#     NAS-SR
#     NSGA-II-NAS-SR
#     Two-Level NSGA-II-NAS-SR
#     Fisher-Guided NAS-SR
#
# Important:
#     All operations preserve spatial resolution.
#
# Input:
#     [B, C, H, W]
#
# Output:
#     [B, C, H, W]
#
# The spatial upsampling is performed only in the
# reconstruction / upsampling module using PixelShuffle.
# ============================================================


import random as rd

import torch
import torch.nn as nn


# ============================================================
# Basic Operations
# ============================================================


class Identity(nn.Module):
    """
    Identity / Skip Connection.

    Used to preserve information and provide a low-cost
    path between nodes.
    """

    def __init__(self):

        super(
            Identity,
            self
        ).__init__()


    def forward(
        self,
        x
    ):

        return x


# ============================================================
# Zero Operation
# ============================================================


class Zero(nn.Module):
    """
    Zero operation.

    Instead of reducing spatial resolution, this operation
    simply removes the feature contribution.
    """

    def __init__(
        self
    ):

        super(
            Zero,
            self
        ).__init__()


    def forward(
        self,
        x
    ):

        return x.mul(
            0.0
        )


# ============================================================
# ReLU + Conv + BN
# ============================================================


class ReLUConvBN(
    nn.Module
):

    def __init__(

        self,

        C_in,

        C_out,

        kernel_size,

        stride=1,

        padding=None,

        affine=True

    ):

        super(
            ReLUConvBN,
            self
        ).__init__()


        if padding is None:

            padding = (

                kernel_size
                -
                1
            ) // 2


        self.op = nn.Sequential(

            nn.ReLU(

                inplace=False
            ),

            nn.Conv2d(

                C_in,

                C_out,

                kernel_size,

                stride=stride,

                padding=padding,

                bias=False
            ),

            nn.BatchNorm2d(

                C_out,

                affine=affine
            )
        )


    def forward(

        self,

        x

    ):

        return self.op(
            x
        )


# ============================================================
# Standard Convolution
# ============================================================


class ConvBNReLU(
    nn.Module
):

    def __init__(

        self,

        C_in,

        C_out,

        kernel_size=3,

        stride=1,

        padding=None,

        affine=True

    ):

        super(
            ConvBNReLU,
            self
        ).__init__()


        if padding is None:

            padding = (

                kernel_size
                -
                1
            ) // 2


        self.op = nn.Sequential(

            nn.Conv2d(

                C_in,

                C_out,

                kernel_size,

                stride=stride,

                padding=padding,

                bias=False
            ),

            nn.BatchNorm2d(

                C_out,

                affine=affine
            ),

            nn.ReLU(

                inplace=True
            )
        )


    def forward(

        self,

        x

    ):

        return self.op(
            x
        )


# ============================================================
# 3x3 Convolution
# ============================================================


class Conv3x3(
    nn.Module
):

    def __init__(

        self,

        C_in,

        C_out,

        affine=True

    ):

        super(
            Conv3x3,
            self
        ).__init__()


        self.op = nn.Sequential(

            nn.Conv2d(

                C_in,

                C_out,

                kernel_size=3,

                stride=1,

                padding=1,

                bias=False
            ),

            nn.BatchNorm2d(

                C_out,

                affine=affine
            ),

            nn.ReLU(

                inplace=True
            )
        )


    def forward(

        self,

        x

    ):

        return self.op(
            x
        )


# ============================================================
# 5x5 Convolution
# ============================================================


class Conv5x5(
    nn.Module
):

    def __init__(

        self,

        C_in,

        C_out,

        affine=True

    ):

        super(
            Conv5x5,
            self
        ).__init__()


        self.op = nn.Sequential(

            nn.Conv2d(

                C_in,

                C_out,

                kernel_size=5,

                stride=1,

                padding=2,

                bias=False
            ),

            nn.BatchNorm2d(

                C_out,

                affine=affine
            ),

            nn.ReLU(

                inplace=True
            )
        )


    def forward(

        self,

        x

    ):

        return self.op(
            x
        )


# ============================================================
# 7x7 Convolution
# ============================================================


class Conv7x7(
    nn.Module
):

    def __init__(

        self,

        C_in,

        C_out,

        affine=True

    ):

        super(
            Conv7x7,
            self
        ).__init__()


        self.op = nn.Sequential(

            nn.Conv2d(

                C_in,

                C_out,

                kernel_size=7,

                stride=1,

                padding=3,

                bias=False
            ),

            nn.BatchNorm2d(

                C_out,

                affine=affine
            ),

            nn.ReLU(

                inplace=True
            )
        )


    def forward(

        self,

        x

    ):

        return self.op(
            x
        )


# ============================================================
# Depthwise Separable Convolution
# ============================================================


class SepConv(
    nn.Module
):

    def __init__(

        self,

        C_in,

        C_out,

        kernel_size,

        padding,

        affine=True

    ):

        super(
            SepConv,
            self
        ).__init__()


        self.op = nn.Sequential(

            nn.ReLU(

                inplace=False
            ),

            # Depthwise convolution

            nn.Conv2d(

                C_in,

                C_in,

                kernel_size=

                kernel_size,

                stride=1,

                padding=padding,

                groups=C_in,

                bias=False
            ),

            # Pointwise convolution

            nn.Conv2d(

                C_in,

                C_out,

                kernel_size=1,

                padding=0,

                bias=False
            ),

            nn.BatchNorm2d(

                C_out,

                affine=affine
            ),

            nn.ReLU(

                inplace=False
            ),

            # Second depthwise convolution

            nn.Conv2d(

                C_out,

                C_out,

                kernel_size=

                kernel_size,

                stride=1,

                padding=padding,

                groups=C_out,

                bias=False
            ),

            # Second pointwise convolution

            nn.Conv2d(

                C_out,

                C_out,

                kernel_size=1,

                padding=0,

                bias=False
            ),

            nn.BatchNorm2d(

                C_out,

                affine=affine
            )
        )


    def forward(

        self,

        x

    ):

        return self.op(
            x
        )


# ============================================================
# Dilated Convolution
# ============================================================


class DilConv(
    nn.Module
):

    def __init__(

        self,

        C_in,

        C_out,

        kernel_size,

        padding,

        dilation,

        affine=True

    ):

        super(
            DilConv,
            self
        ).__init__()


        self.op = nn.Sequential(

            nn.ReLU(

                inplace=False
            ),

            nn.Conv2d(

                C_in,

                C_in,

                kernel_size=

                kernel_size,

                stride=1,

                padding=padding,

                dilation=dilation,

                groups=C_in,

                bias=False
            ),

            nn.Conv2d(

                C_in,

                C_out,

                kernel_size=1,

                padding=0,

                bias=False
            ),

            nn.BatchNorm2d(

                C_out,

                affine=affine
            )
        )


    def forward(

        self,

        x

    ):

        return self.op(
            x
        )


# ============================================================
# Residual Block
# ============================================================


class ResidualBlock(
    nn.Module
):

    def __init__(

        self,

        C_in,

        C_out,

        affine=True

    ):

        super(
            ResidualBlock,
            self
        ).__init__()


        self.conv1 = nn.Conv2d(

            C_in,

            C_out,

            kernel_size=3,

            stride=1,

            padding=1,

            bias=False
        )


        self.bn1 = nn.BatchNorm2d(

            C_out,

            affine=affine
        )


        self.conv2 = nn.Conv2d(

            C_out,

            C_out,

            kernel_size=3,

            stride=1,

            padding=1,

            bias=False
        )


        self.bn2 = nn.BatchNorm2d(

            C_out,

            affine=affine
        )


        if C_in != C_out:

            self.skip = nn.Conv2d(

                C_in,

                C_out,

                kernel_size=1,

                stride=1,

                padding=0,

                bias=False
            )

        else:

            self.skip = Identity()


    def forward(

        self,

        x

    ):

        residual = self.skip(
            x
        )


        out = self.conv1(
            x
        )


        out = self.bn1(
            out
        )


        out = torch.relu(

            out
        )


        out = self.conv2(
            out
        )


        out = self.bn2(
            out
        )


        out = (

            out
            +
            residual
        )


        return torch.relu(

            out
        )


# ============================================================
# Residual Dense Block
# ============================================================


class ResidualDenseBlock(
    nn.Module
):

    def __init__(

        self,

        C,

        growth=32

    ):

        super(
            ResidualDenseBlock,
            self
        ).__init__()


        self.conv1 = nn.Conv2d(

            C,

            growth,

            3,

            1,

            1
        )


        self.conv2 = nn.Conv2d(

            C + growth,

            growth,

            3,

            1,

            1
        )


        self.conv3 = nn.Conv2d(

            C + 2 * growth,

            growth,

            3,

            1,

            1
        )


        self.conv4 = nn.Conv2d(

            C + 3 * growth,

            growth,

            3,

            1,

            1
        )


        self.conv5 = nn.Conv2d(

            C + 4 * growth,

            C,

            3,

            1,

            1
        )


    def forward(

        self,

        x

    ):

        x1 = torch.relu(

            self.conv1(
                x
            )
        )


        x2 = torch.relu(

            self.conv2(

                torch.cat(

                    [
                        x,
                        x1
                    ],

                    dim=1
                )
            )
        )


        x3 = torch.relu(

            self.conv3(

                torch.cat(

                    [
                        x,
                        x1,
                        x2
                    ],

                    dim=1
                )
            )
        )


        x4 = torch.relu(

            self.conv4(

                torch.cat(

                    [
                        x,
                        x1,
                        x2,
                        x3
                    ],

                    dim=1
                )
            )
        )


        x5 = self.conv5(

            torch.cat(

                [
                    x,
                    x1,
                    x2,
                    x3,
                    x4
                ],

                dim=1
            )
        )


        return (

            x
            +
            0.2
            *
            x5
        )


# ============================================================
# Attention Block
# ============================================================


class ChannelAttention(
    nn.Module
):

    def __init__(

        self,

        C,

        reduction=16

    ):

        super(
            ChannelAttention,
            self
        ).__init__()


        hidden = max(

            C // reduction,

            1
        )


        self.attention = nn.Sequential(

            nn.AdaptiveAvgPool2d(

                1
            ),

            nn.Conv2d(

                C,

                hidden,

                1
            ),

            nn.ReLU(

                inplace=True
            ),

            nn.Conv2d(

                hidden,

                C,

                1
            ),

            nn.Sigmoid()
        )


    def forward(

        self,

        x

    ):

        weight = self.attention(
            x
        )


        return (

            x
            *
            weight
        )


# ============================================================
# Large Kernel / Asymmetric Convolution
# ============================================================


class AsymmetricConv(
    nn.Module
):

    def __init__(

        self,

        C

    ):

        super(
            AsymmetricConv,
            self
        ).__init__()


        self.op = nn.Sequential(

            nn.Conv2d(

                C,

                C,

                kernel_size=(1, 7),

                stride=1,

                padding=(0, 3),

                bias=False
            ),

            nn.Conv2d(

                C,

                C,

                kernel_size=(7, 1),

                stride=1,

                padding=(3, 0),

                bias=False
            ),

            nn.BatchNorm2d(

                C
            ),

            nn.ReLU(

                inplace=True
            )
        )


    def forward(

        self,

        x

    ):

        return self.op(
            x
        )


# ============================================================
# Search Space
#
# IMPORTANT:
#
# Every operation keeps:
#
#     H_out = H_in
#     W_out = W_in
#
# This is required for SR cell-level NAS.
# ============================================================


OPS = {

    "none":

        lambda C, affine=True:

        Zero(),


    "skip_connect":

        lambda C, affine=True:

        Identity(),


    "conv_3x3":

        lambda C, affine=True:

        Conv3x3(

            C,

            C,

            affine
        ),


    "conv_5x5":

        lambda C, affine=True:

        Conv5x5(

            C,

            C,

            affine
        ),


    "conv_7x7":

        lambda C, affine=True:

        Conv7x7(

            C,

            C,

            affine
        ),


    "sep_conv_3x3":

        lambda C, affine=True:

        SepConv(

            C,

            C,

            3,

            1,

            affine
        ),


    "sep_conv_5x5":

        lambda C, affine=True:

        SepConv(

            C,

            C,

            5,

            2,

            affine
        ),


    "sep_conv_7x7":

        lambda C, affine=True:

        SepConv(

            C,

            C,

            7,

            3,

            affine
        ),


    "dil_conv_3x3":

        lambda C, affine=True:

        DilConv(

            C,

            C,

            3,

            2,

            2,

            affine
        ),


    "dil_conv_5x5":

        lambda C, affine=True:

        DilConv(

            C,

            C,

            5,

            4,

            2,

            affine
        ),


    "residual_block":

        lambda C, affine=True:

        ResidualBlock(

            C,

            C,

            affine
        ),


    "residual_dense_block":

        lambda C, affine=True:

        ResidualDenseBlock(

            C
        ),


    "channel_attention":

        lambda C, affine=True:

        ChannelAttention(

            C
        ),


    "asymmetric_conv":

        lambda C, affine=True:

        AsymmetricConv(

            C
        )

}


# ============================================================
# Operation Names
# ============================================================


PRIMITIVES = [

    "none",

    "skip_connect",

    "conv_3x3",

    "conv_5x5",

    "conv_7x7",

    "sep_conv_3x3",

    "sep_conv_5x5",

    "sep_conv_7x7",

    "dil_conv_3x3",

    "dil_conv_5x5",

    "residual_block",

    "residual_dense_block",

    "channel_attention",

    "asymmetric_conv"

]


# ============================================================
# Random Operation Selection
#
# Used for:
#     Random NAS initialization
#     PSO initialization
#     NSGA-II population initialization
# ============================================================


def draw_operation(

    num_op=None

):


    if num_op is None:

        num_op = len(

            PRIMITIVES
        )


    index = rd.randint(

        0,

        num_op - 1
    )


    return PRIMITIVES[

        index
    ]


# ============================================================
# Create Operation
# ============================================================


def build_operation(

    op_name,

    C,

    affine=True

):


    if op_name not in OPS:

        raise ValueError(

            "Unknown operation: "

            +
            str(
                op_name
            )
        )


    return OPS[

        op_name

    ](

        C,

        affine
    )


# ============================================================
# Test All Operations
# ============================================================


if __name__ == "__main__":


    print(

        "Testing DIV2K-SR Search Space..."
    )


    C = 64


    H = 48


    W = 48


    x = torch.randn(

        2,

        C,

        H,

        W
    )


    print(

        "\nInput shape:",

        x.shape
    )


    for op_name in PRIMITIVES:


        print(

            "\nTesting operation:",

            op_name
        )


        operation = build_operation(

            op_name,

            C
        )


        y = operation(

            x
        )


        print(

            "Output shape:",

            y.shape
        )


        assert (

            y.shape
            ==
            x.shape
        ), (

            "Spatial or channel dimensions changed "
            "for operation: "
            +
            op_name
        )


    print(

        "\nAll SR operations passed successfully."
    )


    print(

        "\nNumber of candidate operations:",

        len(
            PRIMITIVES
        )
    )


    print(

        "\nSearch Space:"
    )


    for i, op in enumerate(

        PRIMITIVES
    ):

        print(

            i,

            "->",

            op
        )
```
