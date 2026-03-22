import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader():
    def __init__(self):
        self.file_path = './penguins.csv'
        self.data = pd.read_csv(self.file_path)
        self.species_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.data['Species'] = self.species_encoder.fit_transform(self.data['Species'])
        self.data['OriginLocation'] = self.location_encoder.fit_transform(self.data['OriginLocation'])
        self.means = {}
        self.stds = {}

        for col in self.data:
            if col != 'Species' and col != 'OriginLocation':
                mean = self.data[col].mean()
                std = self.data[col].std()
                self.data[col] = (self.data[col] - mean) / std
                self.means[col] = mean
                self.stds[col] = std
    
    def df_to_pairs(self, dataframe):
        pairs = []
        for idx, row in dataframe.iterrows():
            features = row.drop('Species').values
            species = row['Species']
            pairs.append((features, species))
        return pairs

    def loadData(self):
        self.data = self.data.sample(frac = 1, random_state=42).reset_index(drop = True)

        train_data = []
        test_data = []

        for species, group in self.data.groupby('Species', sort=False):
            group.fillna(group.mean(), inplace=True)
            train_data.append(group.iloc[:30])
            test_data.append(group.iloc[30:])
        
        train = pd.concat(train_data, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        test = pd.concat(test_data, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        return (self.df_to_pairs(train), self.df_to_pairs(test))