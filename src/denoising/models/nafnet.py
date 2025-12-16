import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for 4D tensors (N, C, H, W).

    Implements LayerNorm over the channel dimension (C) for image tensors.
    Uses PyTorch's native ``F.layer_norm`` for ONNX/TensorRT compatibility.

    Note:
        This is a simplified, export-friendly alternative to the custom autograd
        function used in the original NAFNet implementation. It produces nearly
        identical results while guaranteeing compatibility with ONNX and TensorRT.

    Example:
        >>> x = torch.randn(2, 64, 32, 32)
        >>> layer_norm = LayerNorm2d(64)
        >>> y = layer_norm(x)  # y.shape == x.shape
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization over channel dimension.

        Converts (N, C, H, W) → (N, H, W, C), applies LayerNorm, then back.

        Args:
            x (torch.Tensor): Input tensor, shape `[N, C, H, W]`.

        Returns:
            torch.Tensor: Normalized tensor, same shape as input.
        """
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            normalized_shape=(self.weight.shape[0],),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        ).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """Simplified Gating Mechanism from NAFNet (Nonlinear Activation-Free Networks).

    Splits the input channel-wise and multiplies the two halves:
    ```
    x1, x2 = x.chunk(2, dim=1)
    return x1 * x2
    ```

    Replaces ReLU/GELU in deep residual blocks while preserving representational
    capacity and improving gradient flow.

    Example:
        >>> x = torch.randn(1, 64, 32, 32)
        >>> gate = SimpleGate()
        >>> y = gate(x)  # y.shape == [1, 32, 32, 32]
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply simple gating.

        Args:
            x (torch.Tensor): Input tensor with even number of channels.

        Returns:
            torch.Tensor: Gated output, shape `[N, C/2, H, W]`.
        """
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    r"""Nonlinear Activation-Free Residual Block (core unit of NAFNet).

    Based on the architecture proposed in:
        Chen et al., Simple Baselines for Image Restoration, arXiv:2204.04676

    Args:
        in_channels (int): Number of input/output channels.
        dw_expand (int, optional): Channel expansion factor for depth-wise conv.
            Defaults to 2.
        ffn_expand (int, optional): Channel expansion for feed-forward network.
            Defaults to 2.
        drop_out_rate (float, optional): Dropout probability. Defaults to 0.0.

    Example:
        >>> block = NAFBlock(64, dw_expand=2, ffn_expand=2)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> y = block(x)  # y.shape == x.shape
    """

    def __init__(
        self, in_channels: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0
    ) -> None:
        super().__init__()
        dw_channel = in_channels * dw_expand

        self.conv1 = nn.Conv2d(in_channels, dw_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, in_channels, 1, 1, 0)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0),
        )
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * in_channels
        self.conv4 = nn.Conv2d(in_channels, ffn_channel, 1, 1, 0)
        self.conv5 = nn.Conv2d(ffn_channel // 2, in_channels, 1, 1, 0)

        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass with dual residual connections.

        Args:
            inp (torch.Tensor): Input feature map, shape `[N, C, H, W]`.

        Returns:
            torch.Tensor: Output feature map, same shape as input.
        """
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    """Nonlinear Activation-Free Network for single-image restoration.

    U-Net style encoder-decoder architecture composed entirely of NAFBlocks.
    Designed for image denoising, deblurring, and super-resolution.

    References:
        - Official code: https://github.com/megvii-research/NAFNet
     Args:
        img_channel (int, optional): Number of input/output image channels.
            Defaults to 3 (RGB).
        width (int, optional): Base channel width (number of channels after intro conv).
            Defaults to 32.
        middle_blk_num (int, optional): Number of NAFBlocks in bottleneck.
            Defaults to 12.
        enc_blk_nums (list[int] | None, optional): Number of NAFBlocks per encoder stage.
            Defaults to `[1, 1, 1, 28]` (NAFNet-S).
        dec_blk_nums (list[int] | None, optional): Number of NAFBlocks per decoder stage.
            Defaults to `[1, 1, 1, 1]`.

    Shape:
        - Input: `[N, C, H, W]`
        - Output: `[N, C, H, W]` (same as input)

    Example:
        >>> model = NAFNet(width=32, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
        >>> x = torch.randn(1, 3, 512, 512)
        >>> y = model(x)  # y.shape == x.shape
        >>> assert torch.allclose(x, y, atol=1e-6)  # only if no noise
    """

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 12,
        enc_blk_nums: list = None,
        dec_blk_nums: list = None,
    ) -> None:
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [1, 1, 1, 28]
        if dec_blk_nums is None:
            dec_blk_nums = [1, 1, 1, 1]

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        inp_padded = F.pad(inp, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")

        x = self.intro(inp_padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs, strict=True):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1], strict=True):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp_padded

        # Crop back
        x = x[..., :h, :w]
        return x
