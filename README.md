# lighting_flash_demo
Minimal Lighting Flash demo

### Install

install pytorch for your os and cuda version

```commandline
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

install lighning-flash and wandb

```commandline
pip3 install lightning-flash
pip3 install wandb
```

create account on http://wandb.com, create token and login using token
```commandline
wandb login
```

### Train

```commandline
python3 train.py
```

Best checkpoints will be saved to the checkpoints directory

to resume from checkpoint

```commandline
python3 train.py --resume checkpoints/woven-dream-5/epoch=2-step=90.ckpt
```

Here's how it will look in wandb

![Image](resources/wadb.png)