class NAFUNetConfig:
    """
    NAFUNet 模型配置类
    """

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 1,
        encoder_channels: list | None = None,
        decoder_channels: list | None = None,
        ignore_background: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels or [2048, 1024, 512, 256, 64]
        self.decoder_channels = decoder_channels or [512, 256, 128, 64]
        self.ignore_background = ignore_background

    @classmethod
    def naf_unet_b(cls, num_classes: int = 9) -> "NAFUNetConfig":
        """
        NAF-UNet-B 配置
        """
        return cls(
            encoder_channels=[1024, 768, 512, 384, 96],
            decoder_channels=[768, 512, 384, 256, 96],
            num_classes=num_classes,
        )

    @classmethod
    def naf_unet_s(cls, num_classes: int = 9) -> "NAFUNetConfig":
        """
        NAF-UNet-S 配置
        """
        return cls(
            encoder_channels=[512, 384, 256, 192, 64],
            decoder_channels=[384, 256, 192, 128, 64],
            num_classes=num_classes,
        )