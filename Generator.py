class ResidualBlock(nn.Module):
    def __init__(self,  channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(256, 256, padding_mode="reflect", kernel_size = 3, padding =1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, padding_mode="reflect", kernel_size = 3, padding =1),
            nn.InstanceNorm2d(256),
            nn.Identity(),
        )

    def forward(self, x):
        return x + self.block(x)


class generator(nn.Module):
    def __init__(self, num_residuals=9):
        super().__init__()

        #encoder
        self.encoder = nn.ModuleList(
          [nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode="reflect",),
          nn.InstanceNorm2d(128),
          nn.ReLU(inplace=True),
          nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode="reflect",),
          nn.InstanceNorm2d(256),
          nn.ReLU(inplace=True),]
          )

        self.residual = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residuals)]
        )

        #decoder
        self.decoder = nn.ModuleList(
           [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),]
           )

        self.last = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


     # forward method
    def forward(self, x):
        # encoding
        for layer in self.encoder:
            x = layer(x)
        #residual
        x = self.residual(x)
        # decoding

        for layer in self.decoder:
            x = layer(x)

        return torch.tanh(self.last(x))

#print out generator details
x = torch.randn((2, 3, 256, 256))
G = generator().to(device)
summary(G,(3, 256, 256))
gen = generator()
print(gen(x).shape)