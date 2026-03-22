import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class Data:
    def __init__(self, data_dir='./data', batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.class_mapping = self._load_class_mapping()
        
    def _load_class_mapping(self):
        mapping = {}
        mapping_file = os.path.join(self.data_dir, 'class_mapping.txt')
        
        with open(mapping_file, 'r') as f:
            for line in f:
                class_name, class_idx = line.strip().split(': ')
                mapping[class_name] = int(class_idx)
        
        return mapping
    
    def load_train_data(self):
        train_dir = os.path.join(self.data_dir, 'train')
        train_dataset = ImageDataset(train_dir, self.class_mapping, is_train=True)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  
            num_workers=2  
        )
        
        return train_loader
    
    def load_test_data(self):
        test_dir = os.path.join(self.data_dir, 'test')
        test_dataset = ImageDataset(test_dir, self.class_mapping, is_train=False)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return test_loader


class ImageDataset(Dataset):
    
    def __init__(self, root_dir, class_mapping, is_train=True):
        self.root_dir = root_dir
        self.class_mapping = class_mapping
        self.is_train = is_train
        self.samples = self._load_samples()
        self.transform = self._get_transforms()
        
    def _load_samples(self):
        samples = []
        
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            
            if not os.path.isdir(class_dir):
                continue
            
            if class_name in self.class_mapping:
                label = self.class_mapping[class_name]
            else:
                continue  
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    samples.append((img_path, label))
        
        return samples
    
    def _get_transforms(self):
        transform_list = [
            PadToSquare(),  
            transforms.Resize((128, 128)),  
            EnsureRGB(),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ]
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path)
        
        image = self.transform(image)
        
        return image, label


class PadToSquare:
    
    def __call__(self, img):
        width, height = img.size
        max_dim = max(width, height)
        
        pad_left = (max_dim - width) // 2
        pad_right = max_dim - width - pad_left
        pad_top = (max_dim - height) // 2
        pad_bottom = max_dim - height - pad_top
        
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        padded_img = transforms.functional.pad(img, padding, fill=0)
        
        return padded_img


class EnsureRGB:
    
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img