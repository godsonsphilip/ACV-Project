import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Two 3x3 Conv2d - BatchNorm2d - ReLU operations with shape assertions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        assert out.shape[1] == self.block[0].out_channels
        return out

class DecoderStage(nn.Module):
    """Single Decoder Stage supporting both Bottleneck-Only and Skip-Concatenation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_skip: bool = False,
        skip_channels: int = 0,
    ):
        super().__init__()
        self.use_skip = use_skip
        # 1x1 conv after bilinear upsampling to adapt feature channels
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        conv_in_ch = (out_channels + skip_channels) if use_skip else out_channels
        self.conv_block = ConvBlock(conv_in_ch, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Upsample by factor of 2 with bilinear interpolation
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.up_conv(x)

        if self.use_skip:
            assert skip is not None, "Skip connection required when use_skip=True"
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv_block(x)
        return x

class Segmenter(nn.Module):
    """Compact Encoder-Decoder Semantic Segmentation Architecture.

    Supports:
        Model A: Baseline compact encoder-decoder with NO skip connections (use_skips=False)
        Model B: Controlled change with skip fusion via concatenation (use_skips=True)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 12,
        encoder_channels: Tuple[int, ...] = (32, 64, 128, 256),
        bottleneck_channels: int = 512,
        use_skips: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.use_skips = use_skips
        self.num_classes = num_classes
        self.model_name = name or ("Model_B_Skips" if use_skips else "Model_A_Baseline")

        # 4 Encoder Downsampling Stages
        c1, c2, c3, c4 = encoder_channels
        self.enc1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(c3, c4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(c4, bottleneck_channels)

        # 4 Decoder Upsampling Stages
        self.dec4 = DecoderStage(
            in_channels=bottleneck_channels,
            out_channels=c4,
            use_skip=use_skips,
            skip_channels=c4 if use_skips else 0,
        )
        self.dec3 = DecoderStage(
            in_channels=c4,
            out_channels=c3,
            use_skip=use_skips,
            skip_channels=c3 if use_skips else 0,
        )
        self.dec2 = DecoderStage(
            in_channels=c3,
            out_channels=c2,
            use_skip=use_skips,
            skip_channels=c2 if use_skips else 0,
        )
        self.dec1 = DecoderStage(
            in_channels=c2,
            out_channels=c1,
            use_skip=use_skips,
            skip_channels=c1 if use_skips else 0,
        )

        # Final Classifier Head
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for conv layers, constant for BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[2:]

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.dec4(b, e4 if self.use_skips else None)
        d3 = self.dec3(d4, e3 if self.use_skips else None)
        d2 = self.dec2(d3, e2 if self.use_skips else None)
        d1 = self.dec1(d2, e1 if self.use_skips else None)

        logits = self.head(d1)
        assert logits.shape[2:] == input_shape, (
            f"Output shape mismatch: {logits.shape[2:]} vs input {input_shape}"
        )
        return logits

def get_model_a(config: dict) -> Segmenter:
    enc_channels = tuple(config["model"]["encoder_channels"])
    bottleneck_ch = config["model"]["bottleneck_channels"]
    num_classes = config["dataset"]["num_classes"]
    return Segmenter(
        in_channels=config["model"]["in_channels"],
        num_classes=num_classes,
        encoder_channels=enc_channels,
        bottleneck_channels=bottleneck_ch,
        use_skips=False,
        name="Model_A_Baseline",
    )

def get_model_b(config: dict) -> Segmenter:
    enc_channels = tuple(config["model"]["encoder_channels"])
    bottleneck_ch = config["model"]["bottleneck_channels"]
    num_classes = config["dataset"]["num_classes"]
    return Segmenter(
        in_channels=config["model"]["in_channels"],
        num_classes=num_classes,
        encoder_channels=enc_channels,
        bottleneck_channels=bottleneck_ch,
        use_skips=True,
        name="Model_B_Skips",
    )

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

