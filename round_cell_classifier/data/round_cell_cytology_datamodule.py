"""Round Cell Classification DataModule"""
import argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from round_cell_classifier.data.image_dataset import ImageDataset

# NUM_WORKERS = 4


class RoundCellCytology(pl.LightningDataModule):
    """
    Round Cell Classification DataModule.
    """

    def __init__(self, batch_size: int, data_dir: str, label_filename: str) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dims = (3, 32, 32)
        self.output_dims = (1,)
        self.mapping = list(range(6))
        self.label_filename = label_filename
        # self.num_workers = NUM_WORKERS
        self.on_gpu = True if torch.cuda.device_count() > 0 else False

    def prepare_data(self, *args, **kwargs) -> None:
        """Load Round Cell data into ImageDataset Class."""
        self.data = ImageDataset(self.label_filename, self.data_dir, self.transform)

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        self.data_train, self.data_val, self.data_test = random_split(self.data, [round(len(self.data)*0.7), round(len(self.data)*0.2), round(len(self.data)*0.1)])  # type: ignore

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            # num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            # num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            # num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
    
    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}