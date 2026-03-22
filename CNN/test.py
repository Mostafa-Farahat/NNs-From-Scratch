from train import TrainEval
import sys
from data import PadToSquare, EnsureRGB
from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

train_eval = TrainEval()
train_eval.load_model()
# image_path = sys.argv[1]

mapping = {
    "asian": 0,
    "boho": 1,
    "coastal": 2,
    "contemporary": 3,
    "craftsman": 4,
    "eclectic": 5,
    "farmhouse": 6,
    "french-country": 7,
    "industrial": 8,
    "mediterranean": 9,
    "minimalist": 10,
    "modern": 11,
    "scandinavian": 12,
    "shabby-chic-style": 13,
    "southwestern": 14,
    "tropical": 15,
    "victorian": 16
}


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(test_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        self.image_files.sort()  # Keep consistent ordering
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, img_name
def generate_predictions(model, test_dir, output_csv='test_predictions.csv', device='cuda'):
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TestDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model.eval()
    model.to(device)
    
    predictions = []
    image_names = []
    
    print(f"Processing {len(test_dataset)} images...")
    
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            image_names.extend(names)
    
    results_df = pd.DataFrame({
        'ImageName': image_names,
        'ClassLabel': predictions
    })
    
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    print(f"Total images processed: {len(results_df)}")
    
    return results_df


predictions_df = generate_predictions(
    model=train_eval.model,
    test_dir='./data/test',
    output_csv='test_predictions.csv',
    device='cpu'
)