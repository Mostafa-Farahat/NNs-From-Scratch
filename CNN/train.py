from torch import optim
import torch
from torch import nn
from network import Network
from torchmetrics import Accuracy

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics import Accuracy
from tqdm import tqdm

class TrainEval():
    def train(self, train_data, epochs, lr):
        self.model = Network(in_channels=3, num_classes=17).to('cpu')
        self.loss = nn.CrossEntropyLoss()
        self.sgd = optim.SGD(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            print(f"epoch {epoch+1} / {epochs}")
            
            running_loss = 0.0
            num_batches = 0

            for batch_index, (x, y) in enumerate(tqdm(train_data)):
                x = x.to('cpu')
                y = y.to('cpu')
                
                preds = self.model(x)
                error = self.loss(preds, y)
                
                self.sgd.zero_grad()
                error.backward()
                self.sgd.step()
                
                running_loss += error.item()
                num_batches += 1
            
            avg_loss = running_loss / num_batches
            print(f"Average loss: {avg_loss:.4f}")

    def save_model(self, filepath='model_weights.pth'):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model weights saved to {filepath}")
    
    def load_model(self, filepath='model_weights.pth'):
        self.model = Network(in_channels=3, num_classes=17).to('cpu')        
        self.model.load_state_dict(torch.load(filepath, map_location='cpu'))
        self.model.eval()
        print(f"Model loaded from {filepath}")

    def evaluate(self, test_data):
        self.model.eval()
        
        acc = Accuracy(task='multiclass', num_classes=17).to('cpu')
        
        with torch.no_grad():
            for images, labels in test_data:
                images = images.to('cpu')
                labels = labels.to('cpu')
                
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                acc.update(preds, labels)
        
        test_acc = acc.compute()
        print(f"Test accuracy: {test_acc:.4f}")
        
        self.model.train()