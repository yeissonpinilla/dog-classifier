from pathlib import Path
from data import Data
from executor import Executor
from vi_t import ViT
from patch_embedding import PatchEmbedding
from transformer_encoder import TransformerEncoder
import torchvision
import torch
import config
from torchinfo import summary
from random import seed as randomSeed
from plotter import Plotter
from torch import nn
import matplotlib.pyplot as plt


# Simply defining the path to the dataset
current_dir = Path(__file__).resolve().parent 
dataset_dir = current_dir.parent / 'datasets/StanfordDogsDataset/myImages'
print(f"Dataset path: {dataset_dir}")

# Transforming the dataset
transformer = Data()
transformed_data = torchvision.datasets.ImageFolder(dataset_dir, transform=transformer.load_data())
print("-" * 50)
print("The original content of Stanford Dogs Dataset") 
print(f"Number of dog images: {len(transformed_data)}")
print(f"Number of dog breeds: {len(transformed_data.classes)}")
print("-" * 50)

# Splitting the dataset
torch.manual_seed(config.SEED_NO)
train_data, val_data, test_data = torch.utils.data.random_split(transformed_data, [0.7, 0.15, 0.15])
print("Number of training images: ", len(train_data))
print("Number of validation images: ", len(val_data))
print("Number of test images: ", len(test_data))
print("-" * 50)

# Loading the dataset
train_dataloader, val_dataloader, test_dataloader = transformer.data_loader(train_data, val_data, test_data, config.BATCH_SIZE, config.NUM_WORKERS)
print(f"Batch size = {config.BATCH_SIZE}")
print(f"Training data size:   {len(train_dataloader.dataset)}\t= {len(train_dataloader)} batches.")
print(f"Validation data size: {len(val_dataloader.dataset)}\t= {len(val_dataloader)} batches.")
print(f"Test data size:       {len(test_dataloader.dataset)}\t= {len(test_dataloader)} batches.")
print("-" * 50)

# Create instance of ViT
vit = ViT(num_classes=len(transformed_data.classes))
summary(model = vit,
        input_size = (config.BATCH_SIZE, config.CHANNELS, config.IMG_SIZE, config.IMG_SIZE), 
        col_names = ["input_size", "output_size", "num_params", "trainable"],
        col_width = 20,
        row_settings = ["var_names"]
)
print("-" * 50)

optimizer_Adam = torch.optim.Adam(params=vit.parameters(), 
                             lr=3e-3, 
                             betas=(0.9, 0.999), 
                             weight_decay=0.3) 

optimizer_SGD = torch.optim.SGD(params=vit.parameters(), 
                                lr=3e-3,
                                momentum=0.2,
                                weight_decay=0.3) 
                                
optimizer_selected = optimizer_Adam

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
device = torch.device(device)

loss_fn = torch.nn.CrossEntropyLoss()

randomSeed(config.SEED_NO)

# Train the model and save the training results to a dictionary
executor = Executor()
results = executor.trainAll(device=device,
                   model=vit,
                   train_dataloader=train_dataloader,
                   eval_dataloader=val_dataloader,
                   loss_fn=loss_fn,
                   optimizer=optimizer_selected,
                   epochs=config.NUM_EPOCH)

# Plot the training results
plotter = Plotter()
plotter.plotComparison(results["train_loss"], "Training Loss", results["val_loss"], "Validation Loss", "Loss Per Epoch", "Loss", "loss-per-epoch.png")
plotter.plotComparison(results["train_accuracy"], "Training Accuracy", results["val_accuracy"], "Validation Accuracy", "Accuracy Per Epoch", "Accuracy", "accuracy-per-epoch.png")

# Test the model
filename1 = "images/test_samoyed_1.jpeg"
filename2 = "images/test_golden_retriever_1.jpeg"
filename3 = "images/test_collie_1.jpeg"
executor.predictImage(model=vit,
             image_path=filename1,
             label_list=transformed_data.classes,
             transform=transformer.load_data(),
             device=device,
             file_name="test_samoyed.jpeg")

executor.predictImage(model=vit,
             image_path=filename2,
             label_list=transformed_data.classes,
             transform=transformer.load_data(),
             device=device,
             file_name="test_golden_retriever.jpeg")

executor.predictImage(model=vit,
             image_path=filename3,
             label_list=transformed_data.classes,
             transform=transformer.load_data(),
             device=device,
             file_name="test_collie.jpeg")