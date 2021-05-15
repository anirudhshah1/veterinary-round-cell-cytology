import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.classification import F1

class Accuracy(pl.metrics.Accuracy):
    """Accuracy Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Metrics in Pytorch-lightning 1.2+ versions expect preds to be between 0 and 1 else fails with the ValueError:
        "The `preds` should be probabilities, but values were detected outside of [0,1] range."
        This is being tracked as a bug in https://github.com/PyTorchLightning/metrics/issues/60.
        This method just hacks around it by normalizing preds before passing it in.
        Normalized preds are not necessary for accuracy computation as we just care about argmax().
        """
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.nn.functional.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, optimizer='Adam', lr=1e-3, loss='cross_entropy'):
        super().__init__()
        self.model = model
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = lr
        self.loss_fn = getattr(torch.nn.functional, loss)

        self.train_f1 = F1(num_classes=7)
        self.val_f1 = F1(num_classes=7)
        self.test_f1 = F1(num_classes=7)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer, "monitor": "val_loss"}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True)
        self.train_acc(logits, y)
        self.train_f1(logits, y)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=True)
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log("val_f1", self.val_f1, on_step=True, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_f1(logits, y)
        self.log("test_f1", self.test_f1, on_step=True, on_epoch=True)