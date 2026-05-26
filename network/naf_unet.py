import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from network.config.configuration_unet import NAFUNetConfig
from network.resnet_modeling import resnet50


class ChannelContextAggregator(nn.Module):
    """
    Channel Context Aggregator (CCA) module.
    Computes dynamic global channel context by aggregating spatially-aware
    channel statistics, capturing both channel importance and spatial consistency.

    Args:
        dim: Channel dimension of input features.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Learnable channel-wise weight parameter.
        # Double-exponential transformation ensures values in (0,1).
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, k, v):
        """
        Args:
            k: Key matrix of shape [B, T, C]
            v: Value matrix of shape [B, T, C]
        Returns:
            Global channel context vector broadcasted to [B, T, C]
        """
        B, T, C = k.shape
        # Double-exponential transformation: w = exp(-exp(gamma))
        w = torch.exp(-torch.exp(self.gamma))  # [C]

        # Apply channel-wise weights to keys
        w_k = w.unsqueeze(0).unsqueeze(0) * k  # [B, T, C]

        # Compute global channel context: Σ (w ⊙ k_t) ⊙ v_t
        gamma_cca = torch.einsum('btc,btc->bc', w_k, v)  # [B, C]

        # Broadcast to all spatial positions
        return gamma_cca.unsqueeze(1).repeat(1, T, 1)  # [B, T, C]


class GatedChannelSpatialAttention(nn.Module):
    """
    Gated Channel-Spatial Attention (GCSA) module.
    Adaptively recalibrates multi-scale features via three components:
    Linear Projections, Channel Context Aggregator (CCA), and Gated Fusion.

    Args:
        n_embd: Input feature channel dimension.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd

        # Reduction factor ρ = 4
        attn_sz = n_embd // 4

        # Linear projections for key, value, and receptance
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)

        self.key_norm = nn.LayerNorm(attn_sz)

        # Output projection
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        # Channel Context Aggregator
        self.cca = ChannelContextAggregator(attn_sz)

    def forward(self, x):
        """
        Args:
            x: Input feature map of shape [B, H*W, C]
        Returns:
            Refined features of shape [B, H*W, C] via residual connection
        """
        B, T, C = x.size()

        # Linear projections
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        # Channel Context Aggregator
        x_attn = self.cca(k, v)

        # Layer normalization
        x_norm = self.key_norm(x_attn)

        # Gated fusion: r acts as a channel-wise gate
        x_gated = r * x_norm

        # Output projection
        x = self.output(x_gated)

        # Residual connection
        return x


class DualPathScaleTransformation(nn.Module):
    """
    Dual-Path Scale Transformation (DPST) module.
    Enables high-fidelity scale alignment between adjacent-scale features.
    Two variants: DPST-L (upsampling) and DPST-H (downsampling).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (defaults to in_channels).
        mode: 'up' for DPST-L (upsampling), 'down' for DPST-H (downsampling).
        upscale_factor: Factor for upsampling (only used in 'up' mode).
    """

    def __init__(self, in_channels, out_channels=None, mode='up', upscale_factor=2):
        super().__init__()
        self.mode = mode
        self.upscale_factor = upscale_factor
        out_channels = out_channels or in_channels

        # ==================== DPST-L (Upsampling) ====================
        if mode == 'up':
            # Path 1: PixelShuffle - rearranges channel to spatial dimensions
            self.path1 = nn.PixelShuffle(upscale_factor)

            # Path 2: Bilinear interpolation + 1x1 convolution
            self.path2 = nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False)
            self.path2_adj = nn.Conv2d(in_channels, in_channels // (upscale_factor ** 2), kernel_size=1)

            self.shuffle_channels = in_channels // (upscale_factor ** 2)
            fused_channels = self.shuffle_channels

            # Skip connection adapter
            if in_channels != out_channels or upscale_factor > 1:
                self.skip_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False)
                )
            else:
                self.skip_conv = nn.Identity()

        # ==================== DPST-H (Downsampling) ====================
        else:  # mode == 'down'
            # Path 1: Max pooling
            self.path1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

            # Path 2: Strided convolution
            self.path2 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
            fused_channels = in_channels

            # Skip connection adapter
            if in_channels != out_channels or self.path1.stride > 1:
                self.skip_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
                )
            else:
                self.skip_conv = nn.Identity()

        # Depthwise separable convolution for feature refinement
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1, groups=fused_channels),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Dual-path processing
        x1 = self.path1(x)
        x2 = self.path2(x)
        if self.mode == 'up':
            x2 = self.path2_adj(x2)

        # Fuse dual paths
        out = x1 + x2

        # Depthwise separable convolution refinement
        out = self.depthwise_conv(out)

        # Residual connection from input
        residual = self.skip_conv(x)
        out = out + residual

        return out


class NAFDecoderLayer(nn.Module):
    """
    Neighbor-Aware Fusion (NAF) decoder layer.

    This layer aggregates features exclusively from adjacent scales (lower,
    current, and higher). It applies Dual-Path Scale Transformation (DPST)
    for high-fidelity alignment and Gated Channel-Spatial Attention (GCSA)
    for efficient feature refinement.

    Args:
        in_size: Input channel size for concatenated features.
        out_size: Output channel size after convolution blocks.
        high_channels: Channel dimension of higher-scale feature (finer resolution).
            If None, higher-scale fusion is skipped.
        low_channels: Channel dimension of lower-scale feature (coarser resolution).
            If None, lower-scale fusion is skipped.
        upscale_factor: Upsampling factor for lower-scale alignment. Default: 2.
        attn_s: Whether to apply Gated Channel-Spatial Attention. Default: False.
    """

    def __init__(self, in_size, out_size, high_channels=None, low_channels=None,
                 upscale_factor=2, attn_s=False):
        super().__init__()

        self.attn_s = attn_s

        # DPST for lower-scale feature upsampling (DPST-L)
        if low_channels is not None:
            self.low_dpst = DualPathScaleTransformation(
                in_channels=low_channels,
                out_channels=low_channels // 4,
                mode='up',
                upscale_factor=upscale_factor
            )
            in_size += low_channels // 4

        # DPST for higher-scale feature downsampling (DPST-H)
        if high_channels is not None:
            self.high_dpst = DualPathScaleTransformation(
                in_channels=high_channels,
                out_channels=high_channels,
                mode='down'
            )


        # Upsample deeper decoder output
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Gated Channel-Spatial Attention (GCSA)
        self.attn = GatedChannelSpatialAttention(in_size) if attn_s else nn.Identity()

        # Channel compression via 1x1 convolution
        self.compression = nn.Sequential(
            nn.Conv2d(in_size, in_size // 2, 1),
            nn.BatchNorm2d(in_size // 2),
            nn.ReLU(inplace=True)
        )

        # Two consecutive 3x3 convolution blocks
        self.conv1 = nn.Conv2d(in_size // 2, out_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, decoder_in, skip_conn, higher=None, lower=None):
        """
        Forward pass of the NAF decoder layer.

        Args:
            decoder_in: Output from deeper decoder layer (coarser scale).
            skip_conn: Current-scale skip connection from encoder.
            higher: Higher-scale feature (finer resolution) from encoder.
                Used for DPST-H downsampling.
            lower: Lower-scale feature (coarser resolution) from encoder.
                Used for DPST-L upsampling.

        Returns:
            Refined output feature map after neighbor-aware fusion.
        """
        # Upsample deeper decoder output
        upsampled = self.up(decoder_in)

        # Start with current skip connection
        fusion = skip_conn

        # Fuse higher-scale feature (fine details) via DPST-H
        if higher is not None:
            higher_aligned = self.high_dpst(higher)
            fusion = torch.cat([fusion, higher_aligned], 1)

        # Fuse lower-scale feature (semantic context) via DPST-L
        if lower is not None:
            lower_aligned = self.low_dpst(lower)
            fusion = torch.cat([fusion, lower_aligned], 1)

        # Concatenate with upsampled decoder output
        outputs = torch.cat([fusion, upsampled], 1)

        # Apply Gated Channel-Spatial Attention
        if self.attn_s:
            B, C, H, W = outputs.size()
            x_reshaped = outputs.float().view(B, C, -1).permute(0, 2, 1)

            with autocast(device_type='cuda', enabled=False, dtype=torch.float32):
                x_attn = self.attn(x_reshaped)
                x_reshaped = x_reshaped + x_attn

            outputs = x_reshaped.permute(0, 2, 1).view(B, C, H, W).to(fusion.dtype)

        # Channel compression
        outputs = self.compression(outputs)

        # Two consecutive 3x3 convolutions
        outputs = self.conv1(outputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)

        return outputs


class NAFUNetModel(nn.Module):
    """
    NAF-UNet: Neighbor-Aware Multi-Scale Fusion with Channel Attention
    for Medical Image Segmentation.

    Full encoder-decoder architecture with ResNet50 backbone and
    NAF decoder layers that selectively aggregate features from adjacent scales.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        encoder_channels = config.encoder_channels
        decoder_channels = config.decoder_channels

        # ResNet50 encoder backbone
        self.encoder = resnet50(config)

        # NAF decoder layers (from deep to shallow)
        self.decoder_layer4 = NAFDecoderLayer(
            encoder_channels[0] + encoder_channels[1] + encoder_channels[2],
            decoder_channels[0],
            high_channels=encoder_channels[2],
            attn_s=True
        )

        self.decoder_layer3 = NAFDecoderLayer(
            encoder_channels[2] + decoder_channels[0] + encoder_channels[3],
            decoder_channels[1],
            high_channels=encoder_channels[3],
            low_channels=encoder_channels[1],
            attn_s=True
        )

        self.decoder_layer2 = NAFDecoderLayer(
            encoder_channels[3] + decoder_channels[1] + encoder_channels[4],
            decoder_channels[2],
            high_channels=encoder_channels[4],
            low_channels=encoder_channels[2],
            attn_s=True
        )

        self.decoder_layer1 = NAFDecoderLayer(
            encoder_channels[4] + decoder_channels[2],
            decoder_channels[3],
            low_channels=encoder_channels[3],
            attn_s=True
        )

        # Up layer to restore original resolution
        self.up_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(decoder_channels[3], decoder_channels[4], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels[4], decoder_channels[4] // 2, 1),
            nn.ReLU(),
            nn.Conv2d(decoder_channels[4] // 2, config.num_classes, 1)
        )

        # Auxiliary heads for deep supervision
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(decoder_channels[0], config.num_classes, kernel_size=1),
            nn.Conv2d(decoder_channels[1], config.num_classes, kernel_size=1),
            nn.Conv2d(decoder_channels[2], config.num_classes, kernel_size=1),
            nn.Conv2d(decoder_channels[4], config.num_classes, kernel_size=1)
        ])

        # Auxiliary loss weights (shallower layers receive higher weights)
        self.aux_weights = [0.1, 0.3, 0.6, 1.0]

        self._init_weights()

    def _init_weights(self):
        """Initialize convolution and batch normalization weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def cross_entropy_loss(self, pred, target, weight=None, ignore_index=-100):
        """Compute cross-entropy loss."""
        return F.cross_entropy(pred, target, weight=weight, ignore_index=ignore_index)

    def dice_loss(self, pred, target, smooth=1e-6, ignore_background=True):
        """Compute Dice loss with optional background ignoring."""
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        pred_soft = F.softmax(pred, dim=1)

        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        y_sum = target_one_hot.pow(2).sum(dim=(2, 3))
        z_sum = pred_soft.pow(2).sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (z_sum + y_sum + smooth)
        dice_loss_per_class = 1.0 - dice

        if ignore_background:
            mask = torch.ones(num_classes, dtype=torch.bool, device=pred.device)
            mask[0] = False
            return dice_loss_per_class[:, mask].mean()
        return dice_loss_per_class.mean()

    def forward(self, image, labels=None):
        """
        Forward pass of NAF-UNet.

        Args:
            image: Input image of shape [B, C, H, W]
            labels: Ground truth labels for training (optional)
        Returns:
            Dict containing 'logits', 'aux_outputs', and optional 'loss'
        """
        # Encoder feature extraction
        [feat1, feat2, feat3, feat4, feat5] = self.encoder(image)

        # NAF decoder forward pass
        up4 = self.decoder_layer4(
            feat5, feat4, higher=feat3
        )

        up3 = self.decoder_layer3(
            up4, feat3, higher=feat2, lower=feat4
        )

        up2 = self.decoder_layer2(
            up3, feat2, higher=feat1, lower=feat3
        )

        up1 = self.decoder_layer1(
            up2, feat1, lower=feat2
        )

        # Up layer
        up1 = self.up_layer(up1)

        # Final segmentation logits
        logits = self.seg_head(up1)

        # Auxiliary outputs for deep supervision
        aux_outputs = []
        decoder_features = [up4, up3, up2, up1]

        for feat, head in zip(decoder_features, self.aux_heads):
            aux_logits = head(feat)
            if aux_logits.shape[-2:] != logits.shape[-2:]:
                aux_logits = F.interpolate(
                    aux_logits, size=logits.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            aux_outputs.append(aux_logits)

        # Loss computation (training only)
        loss = None
        if labels is not None and self.training:
            if labels.dim() == 4:
                labels = labels.squeeze(1)

            # Main loss: CE + α * Dice (α = 1.2)
            ce_loss = self.cross_entropy_loss(logits, labels)
            dice = self.dice_loss(logits, labels, ignore_background=True)
            main_loss = ce_loss + 1.2 * dice

            # Auxiliary losses
            aux_losses = 0
            for i, aux_logits in enumerate(aux_outputs):
                weight = self.aux_weights[i] if i < len(self.aux_weights) else 0.1
                aux_ce = self.cross_entropy_loss(aux_logits, labels)
                aux_dice = self.dice_loss(aux_logits, labels, ignore_background=True)
                aux_losses += weight * (aux_ce + aux_dice)

            loss = main_loss + aux_losses * 0.2

        return {
            "logits": logits,
            "aux_outputs": aux_outputs,
            "loss": loss
        }


if __name__ == '__main__':

    # NAF-UNet-B configuration
    config = NAFUNetConfig.naf_unet_b(num_classes=9)

    # NAF-UNet-S configuration
    # config = NAFUNetConfig.naf_unet_s(num_classes=9)

    model = NAFUNetModel(config=config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    inputs = torch.randn(1, 1, 224, 224)
    print(f"Input shape: {inputs.shape}")