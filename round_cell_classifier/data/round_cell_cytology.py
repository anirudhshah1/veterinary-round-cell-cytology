"""Round Cell Classification DataModule"""
import argparse
import pytorch_lightning as pl
from torch.utils.data import random_split
from torchvision import transforms, datasets


class RoundCellCytology(pl.LightningDataModule):
    """
    Round Cell Classification DataModule.
    """

    def __init__(self, config, batch_size: int, data_dir: str) -> None:
        super().__init__(args)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dims = (3, 32, 32)
        self.output_dims = (1,)
        self.mapping = list(range(6))

    def prepare_data(self, *args, **kwargs) -> None:
        """Load Round Cell data into ImageFolder Class."""
        self.data = datasets.ImageFolder(root=self.config_train_data_folder, transform=transform)

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        self.data_train, self.data_val, self.data_test = random_split(self.data, [int(len(self.data*0.7)), int(len(self.data*0.2)), int(len(self.data*0.1))])  # type: ignore

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)
    
    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
    return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}