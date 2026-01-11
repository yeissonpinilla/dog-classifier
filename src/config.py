
import os

### Global VARIABLES ###
SEED_NO = 100

# Hyper-parameters configs
NUM_EPOCH = 15

BATCH_SIZE = 16
# os.cpu_count() returns None if undetermined, so we default to 1 in that case
NUM_WORKERS = os.cpu_count() or 1
IMG_SIZE = 224
H = IMG_SIZE
W = IMG_SIZE
# ViT parameters
PATCH_SIZE = 16
CHANNELS = 3
