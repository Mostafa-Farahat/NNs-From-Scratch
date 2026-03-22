from data import Data
from train import TrainEval
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import torch
import pandas as pd

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

data = Data()

test_data = data.load_test_data()
train_data = data.load_train_data()

train_eval = TrainEval()
train_eval.train(train_data, 1, 0.005)
train_eval.evaluate(test_data)
train_eval.save_model()


for filename in os.listdir("./data/test"):
        filepath = os.path.join("./data/test", filename)
        
        if not os.path.isfile(filepath):
            continue
        
        try:
            with Image.open(filepath) as img:
                img.verify()  
            
        except Exception as e:
            os.remove(filepath)

predictions_df = generate_predictions(
    model=train_eval.model,
    test_dir='./data/test',
    output_csv='test_predictions.csv',
    device='cpu'
)

