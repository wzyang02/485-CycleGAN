def normal_init(m, mean, std):

    """
    Helper function. Initialize model parameter with given mean and std.
    """
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        # delete start
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        # delete end


class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.ModuleList(
            [nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True, padding_mode="reflect",),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True, padding_mode="reflect",),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True, padding_mode="reflect",),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True, padding_mode="reflect",),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True, padding_mode="reflect",),
            ]
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return torch.sigmoid(x)

#print out discriminator details
x = torch.randn((2, 3, 256, 256))
D = discriminator().to(device)
summary(D,(3, 256, 256))
disc = discriminator()
print(disc(x).shape)