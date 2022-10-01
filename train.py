import torch
import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier
import pytorch_lightning as pl
import wandb
from torchvision.transforms.functional import to_tensor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import argparse


class WandbImagePredCallback(pl.Callback):
    """Logs the input images and output predictions of a module.
    Predictions and labels are logged as class indices."""

    def __init__(self, num_samples=32):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loader = trainer.datamodule.val_dataloader()
        for i in range(self.num_samples):
            x = to_tensor(val_loader.dataset[i]['input']).unsqueeze(0).to(pl_module.device)
            y = val_loader.dataset[i]['target']
            pred = torch.argmax(pl_module(x), 1)
            trainer.logger.experiment.log({
                "val/examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                ],
                "global_step": trainer.global_step
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    wandb_logger = WandbLogger(project="lightning bolts demo", log_model=False)
    wandb_image_f = WandbImagePredCallback(num_samples=4)

    checkpoint_f = ModelCheckpoint(dirpath=f"checkpoints//{wandb_logger.experiment.name}/",
                                   save_top_k=1, mode='max',
                                   monitor="val_accuracy")

    # Create the DataModule
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "/mnt/data/data")

    datamodule = ImageClassificationData.from_folders(
        train_folder="/mnt/data/data/hymenoptera_data/train/",
        val_folder="/mnt/data/data/hymenoptera_data/val/",
        batch_size=4,
        transform_kwargs={"image_size": (196, 196), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    )

    # Init or load model
    if args.resume is None:
        model = ImageClassifier(backbone="resnet18", labels=datamodule.labels)
    else:
        model = ImageClassifier.load_from_checkpoint(args.resume)

    # Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count(),
                            logger=wandb_logger,
                            enable_checkpointing=True,
                            callbacks=[LearningRateMonitor(), checkpoint_f, wandb_image_f])
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # Predict what's on a few images! ants or bees?
    datamodule = ImageClassificationData.from_files(
        predict_files=[
            "https://pl-flash-data.s3.amazonaws.com/images/ant_1.jpg",
            "https://pl-flash-data.s3.amazonaws.com/images/ant_2.jpg",
            "https://pl-flash-data.s3.amazonaws.com/images/bee_1.jpg",
        ],
        batch_size=3,
    )
    predictions = trainer.predict(model, datamodule=datamodule, output="labels")
    print(predictions)

    # 5. Save the model!
    trainer.save_checkpoint("image_classification_model.pt")

