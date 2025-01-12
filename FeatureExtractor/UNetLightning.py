#Biblioteki i pakiety
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import MulticlassAccuracy
import torchvision.utils as vutils
from PIL import Image

from UNet import UNet

#Eklasa ekstraktora cech
class UNetLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet()
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _, _ = batch

        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            output = self(x)
            loss = self.criterion(output, x)

        self.manual_backward(loss)
        optimizer.step()

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)

        if batch_idx % 200 == 0:
            self.log_images_to_comet(
                self.logger,
                x,
                output,
                epoch=self.current_epoch,
                step=self.global_step,
                name='train_sample'
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        output = self(x)
        loss = self.criterion(output, x)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=True)

        if batch_idx == 0:
            self.log_images_to_comet(
                self.logger,
                x,
                output,
                epoch=self.current_epoch,
                step=self.global_step,
                name='val_sample'
            )

        return loss

    def test_step(self, batch, batch_idx):
        x, _, _ = batch
        output = self(x)
        loss = self.criterion(output, x)

        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=True)

        if batch_idx == 0:
            self.log_images_to_comet(
                self.logger,
                x,
                output,
                epoch=self.current_epoch,
                step=self.global_step,
                name='test_sample'
            )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

            weights_path = f'best_weights_epoch_{self.current_epoch}_val_loss_{val_loss:.4f}.pth'
            torch.save(self.model.state_dict(), weights_path)

            self.logger.experiment.log_model(
                name='best_weights',
                file_or_folder=weights_path,
                overwrite=True
            )
            print(f'Weights saved and logged to CometML: {weights_path}')

    def log_images_to_comet(self, logger, image, output, epoch, step, name='sample'):
        grid = vutils.make_grid(
            [image[0].cpu(), output[0].cpu()],
            nrow=2,
            normalize=True
        )

        grid_np = grid.permute(1, 2, 0).detach().numpy()  # CxHxW -> HxWxC

        grid_image = Image.fromarray((grid_np * 255).clip(0, 255).astype('uint8'))

        logger.experiment.log_image(
            grid_image, name=f'{name}_epoch_{epoch}_step_{step}.png'
        )

    def extract_features(self, x):
        """
        Wywołanie funkcji extract_features z modelu UNet.
        """
        return self.model.extract_features(x)

    def summary(self, input_size=(3, 224, 224)):
        """
        Wyświetla szczegóły modelu i warstw, korzystając z torchinfo.

        :param input_size: Rozmiar danych wejściowych (domyślnie: (3, 224, 224))
        """
        return summary(self.model, input_size=(1, *input_size), col_names=["input_size", "output_size", "num_params", "kernel_size"], depth=3)
    
