import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

class Executor:
    def __init__(self):
        pass

    def train_1Epoch(self, device: torch.device, 
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    loss_fn: nn.Module, 
    optimizer: nn.Module):
        model.train()
        running_loss, running_correct = 0.0, 0.0
        for batch_id, data in enumerate(dataloader):
            X = data[0].to(device)
            y = data[1].to(device)

            optimizer.zero_grad()
            y_predict = model(X)
            loss = loss_fn(y_predict, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            y_predict_label = torch.argmax(y_predict, dim=1)
            running_correct += torch.eq(y, y_predict_label).sum().item()
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = running_correct / len(dataloader.dataset)
        return epoch_loss, epoch_accuracy

    def eval_1Epoch(self, device: torch.device, 
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    loss_fn: nn.Module,
    optimizer: nn.Module):
        model.eval()
        phase = 'Eval'
        running_loss, running_correct = 0.0, 0.0
        completed_samples = 0
        with torch.inference_mode():
            for batch_id, data in enumerate(dataloader):
                X = data[0].to(device)
                y = data[1].to(device)

                optimizer.zero_grad()
                y_predict = model(X)
                loss = loss_fn(y_predict, y)
                running_loss += loss.item()
                y_predict_label = torch.argmax(y_predict, dim=1)
                running_correct += torch.eq(y, y_predict_label).sum().item()
                completed_samples += len(y)
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = running_correct / completed_samples
        return epoch_loss, epoch_accuracy


    def trainAll(self, device: torch.device,
                model: nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                eval_dataloader: torch.utils.data.DataLoader,             
                loss_fn: nn.Module,
                optimizer: nn.Module,
                epochs: int):
        start_train = time.time()
        results = {"train_loss" : [],
                "train_accuracy" : [], 
                "val_loss" : [],
                "val_accuracy" : []}
        
        model.to(device)
        
        for epo in tqdm(range(epochs)):
            train_loss, train_accuracy = self.train_1Epoch(device=device, 
                                                    model=model,
                                                    dataloader=train_dataloader,
                                                    loss_fn=loss_fn,
                                                    optimizer=optimizer)
            val_loss, val_accuracy = self.eval_1Epoch(device=device, 
                                                model=model,
                                                dataloader=eval_dataloader,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer)
            results["train_loss"].append(train_loss)
            results["train_accuracy"].append(train_accuracy)
            results["val_loss"].append(val_loss)
            results["val_accuracy"].append(val_accuracy)
            
            
            print(f"Epoch: {1 + epo}/{epochs}, training: loss = {train_loss:.4f} \t accuracy = {train_accuracy:.3%}")
            print(f"Epoch: {1 + epo}/{epochs}, validation: loss = {val_loss:.4f} \t accuracy = {val_accuracy:.3%}")
        
        end_train = time.time()
        time_spent = (end_train - start_train) / 60
        print(f"Time spent: {time_spent:.2f} minutes")
        return results  

    def randomSeed(seed_number: int):
        torch.manual_seed(seed_number)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_number)

    def predictImage(self, 
                 model: nn.Module, 
                 image_path: str,
                 label_list: list,
                 device: torch.device,
                 transform=None, 
                 file_name: str=""):
        image = torchvision.io.read_image(str(image_path))
        image = image.type(torch.float32)
        image = image / 255.0

        if transform:
            image = transform(image)

        model.to(device)
        model.eval()
        with torch.inference_mode():
            img_tensor = image.unsqueeze(dim=0)
            pred_logit = model(img_tensor.to(device))
            pred_prob = torch.softmax(pred_logit, dim=1)
            max_prob = pred_prob.max()
                    
            pred_id = torch.argmax(pred_prob, dim=1)
            label_str = label_list[pred_id]
            
            print("Prediction probability for each class:",pred_prob)

        plt_img = image.permute(1, 2, 0)
        plt.imshow(plt_img)
        title = f"Prediction: {label_str}\n probility = {max_prob:.2%}"
        plt.title(title)
        plt.axis(False)
        plt.savefig(file_name)