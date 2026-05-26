import os
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import DiceLoss
from torchvision import transforms


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    # ========== 基础准备 ==========
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(str(args))

    # ========== 数据 ==========
    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([
            RandomGenerator(output_size=[args.img_size, args.img_size])
        ])
    )

    logging.info(f"Train samples: {len(db_train)}")

    def worker_init_fn(worker_id):
        seed = args.seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size * args.n_gpu,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # ========== 模型 ==========
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.base_lr,
        epochs=args.max_epochs,
        steps_per_epoch=len(trainloader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
    )

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))

    # ========== 训练 ==========
    iter_num = 0
    max_epoch = args.max_epochs

    for epoch in tqdm(range(max_epoch), ncols=70):
        for sampled_batch in trainloader:
            images = sampled_batch["image"].cuda()
            labels = sampled_batch["label"].cuda()

            outputs = model(
                images,
                labels=labels.long() if args.return_loss else None,
            )

            loss = outputs["loss"]
            logits = outputs["logits"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            iter_num += 1

            writer.add_scalar("info/lr", scheduler.get_last_lr()[0], iter_num)
            writer.add_scalar("info/total_loss", loss.item(), iter_num)

            logging.info(
                "Epoch [%d] Loss: %.4f LR: %.6f",
                epoch, loss.item(), scheduler.get_last_lr()[0]
            )

            if iter_num % 20 == 0:
                vis_img = images[1, 0:1, :, :]
                vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())

                writer.add_image("train/Image", vis_img, iter_num)

                pred = torch.argmax(
                    torch.softmax(logits, dim=1), dim=1, keepdim=True
                )

                writer.add_image("train/Prediction", pred[1, ...] * 50, iter_num)
                writer.add_image(
                    "train/GroundTruth",
                    labels[1, ...].unsqueeze(0) * 50,
                    iter_num
                )

        if epoch > max_epoch // 2 and (epoch + 1) % 50 == 0:
            ckpt = os.path.join(snapshot_path, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            logging.info("Save model: %s", ckpt)

    final_ckpt = os.path.join(snapshot_path, f"epoch_{max_epoch - 1}.pth")
    torch.save(model.state_dict(), final_ckpt)
    logging.info("Final model saved: %s", final_ckpt)

    writer.close()
    return "Training Finished!"