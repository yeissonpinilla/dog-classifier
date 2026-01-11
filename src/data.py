import torch
import matplotlib.pyplot as plt
import config
from PIL import Image
from torchvision.transforms import v2

class Data:
    def __init__(self):
        pass

    def load_data(self):
        transform = v2.Compose([
            v2.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32)
        ])
        return transform

    def data_loader(self, train_data, val_data, test_data, batch_size, num_workers, shuffle=True):
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader