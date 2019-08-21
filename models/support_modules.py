from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, configuration, idx):
        super().__init__()

        # Block Index
        self.index = idx + 1

        # Encoder Block
        self.encoder = nn.Sequential(
            nn.Conv2d(configuration['in_channels'][idx], configuration['out_channels'][idx],
                      kernel_size=configuration['kernel'][idx], stride=configuration['stride'][idx],
                      padding=configuration['padding'][idx], dilation=configuration['dilation'][idx]),
            # output dimensions: 32 * 600 * 20
            nn.ReLU(True),
            nn.BatchNorm2d(configuration['out_channels'][idx]),
            nn.Dropout2d(p=configuration['dropout'])
        )

        if idx != 0:
            # module = nn.Sequential(nn.ReLU(True), nn.BatchNorm2d(configuration['in_channels'][idx]),
            #                       nn.Dropout2d(p=configuration['dropout']))
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(configuration['out_channels'][idx], configuration['in_channels'][idx],
                                   kernel_size=configuration['kernel'][idx], stride=configuration['stride'][idx],
                                   padding=configuration['padding'][idx], dilation=configuration['dilation'][idx]),
                nn.ReLU(True),
                nn.BatchNorm2d(configuration['in_channels'][idx]),
                nn.Dropout2d(p=configuration['dropout'])
            )
        else:
            # module = nn.Sequential(nn.BatchNorm2d(1), nn.Sigmoid())
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(configuration['out_channels'][idx], configuration['in_channels'][idx],
                                   kernel_size=configuration['kernel'][idx], stride=configuration['stride'][idx],
                                   padding=configuration['padding'][idx], dilation=configuration['dilation'][idx]),
                nn.BatchNorm2d(configuration['in_channels'][idx]),
                nn.Sigmoid()
            )

        #     # Decoder Block
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(configuration['out_channels'][idx], configuration['in_channels'][idx],
        #                        kernel_size=configuration['kernel'][idx], stride=configuration['stride'][idx],
        #                        padding=configuration['padding'][idx], dilation=configuration['dilation'][idx]),
        #     module
        # )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, idx, flatten=False, p=0.1):
        super().__init__()

        self.index = idx+1

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(True),
            nn.Dropout(p=p)
        )

        if flatten:
            self.encoder = nn.Sequential(Flatten(), self.encoder)

        self.decoder = nn.Sequential(
            nn.Linear(out_dim, in_dim),
            nn.ReLU(True),
            nn.Dropout(p=p)
        )

        if flatten:
            self.decoder = nn.Sequential(self.decoder, UnFlatten())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], 3200)


class UnFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], 128, 25, 1)
