class Patcher:
    def __init__(self):
        self.num_patches = 0

    def calculate_num_patches(self, height, width, patch_size):
        self.num_patches = int((height * width) / (patch_size * patch_size))
        return self.num_patches

