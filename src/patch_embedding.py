import torch.nn as nn

class PatchEmbedding(nn.Module):

    def __init__(self, channels = 3, image_size = 224, patch_size = 16, embedding_size = 768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.image_size = image_size

        self.divider = nn.Conv2d(in_channels=channels, out_channels=embedding_size, kernel_size=patch_size, stride=patch_size)

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = self.divider(x)
        x = self.flatten(x)
        x = x.transpose(1, 2)
        return x