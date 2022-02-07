from torch.utils.data import Dataset
import pandas as pd
import PIL
import os

class ImageDataset(Dataset):
    def __init__(self, label_filename: str, data_dir: str, transform = None):
        self.labels = pd.read_csv(os.path.join(data_dir, label_filename))
        self.data_dir = data_dir
        self.transform = transform
        self.class_mapper = self.class2index()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        
        filename = self.labels.loc[index, "File"]
        label = self.class_mapper[self.labels.loc[index, "Label"]]
        image = PIL.Image.open(os.path.join(self.data_dir, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class2index(self):
        labels_list = self.labels.Label.unique()
        return {labels_list[i]: i for i in range(len(labels_list))}