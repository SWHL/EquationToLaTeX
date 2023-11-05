# -*- encoding: utf-8 -*-
import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import wandb
from munch import Munch
from tqdm.auto import tqdm

from data.dataset import Equation2LatexDataset
from eval import evaluate
from models import get_model
from utils import (
    get_optimizer,
    get_scheduler,
    gpu_memory_check,
    in_model_path,
    parse_args,
    seed_everything,
    read_yaml,
    write_yaml,
    mkdir,
)

os.environ["OMP_NUM_THREADS"] = "1"


def train(args):
    train_dataset = Equation2LatexDataset(args.data, args.tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        num_workers=1,
        shuffle=True,
        collate_fn=train_dataset.batch_op,
    )

    val_dataset = Equation2LatexDataset(args.valdata, args.tokenizer)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        collate_fn=val_dataset.batch_op,
        num_workers=1,
        shuffle=False,
    )

    device = args.device

    model = get_model(args)
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_memory_check(model, args)

    max_bleu, max_token_acc = 0, 0

    out_path = Path(args.model_path) / args.name
    mkdir(out_path)

    if args.load_chkpt is not None:
        model.load_state_dict(torch.load(args.load_chkpt, map_location=device))

    def save_models(e, bleu, token_acc, step=0):
        save_model_path = (
            out_path
            / f"Epoch{e+1}_step{step:02d}_bleu{bleu:.4f}_tokenacc{token_acc:.4f}.pth"
        )
        torch.save(model.state_dict(), str(save_model_path))

        save_yaml_path = out_path / "config.yaml"
        write_yaml(save_yaml_path, dict(args))

    opt = get_optimizer(args.optimizer)(model.parameters(), args.lr, betas=args.betas)
    scheduler = get_scheduler(args.scheduler)(
        opt, step_size=args.lr_step, gamma=args.gamma
    )

    microbatch = args.get("micro_batchsize", -1)
    if microbatch == -1:
        microbatch = args.batchsize

    try:
        for e in range(args.epoch, args.epochs):
            args.epoch = e
            dset = tqdm(iter(train_dataloader))
            for i, (seq, im) in enumerate(dset):
                if seq is None and im is None:
                    continue

                im = im[0]
                opt.zero_grad()

                total_loss = 0
                for j in range(0, len(im), microbatch):
                    tgt_seq = seq["input_ids"][j : j + microbatch].to(device)
                    tgt_mask = (
                        seq["attention_mask"][j : j + microbatch].bool().to(device)
                    )

                    loss = (
                        model.data_parallel(
                            im[j : j + microbatch].to(device),
                            device_ids=args.gpu_devices,
                            tgt_seq=tgt_seq,
                            mask=tgt_mask,
                        )
                        * microbatch
                        / args.batchsize
                    )

                    loss.backward()
                    total_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                opt.step()
                scheduler.step()
                dset.set_description(
                    f"Epoch {e+1}/{args.epochs} Loss:  {total_loss:.4f}"
                )

                if args.wandb:
                    wandb.log({"train/loss": total_loss})

                if (i + 1 + len(train_dataloader) * e) % args.sample_freq == 0:
                    bleu_score, edit_distance, token_accuracy = evaluate(
                        model,
                        val_dataloader,
                        args,
                        num_batches=int(args.valbatches * e / args.epochs),
                        name="val",
                    )
                    if bleu_score > max_bleu and token_accuracy > max_token_acc:
                        max_bleu, max_token_acc = bleu_score, token_accuracy
                        save_models(e, bleu=max_bleu, token_acc=max_token_acc, step=i)

            # if (e + 1) % args.save_freq == 0:
            #     save_models(e, step=len(dataloader))

            if args.wandb:
                wandb.log({"train/epoch": e + 1})

    except KeyboardInterrupt:
        if e >= 2:
            save_models(e, step=i)
        raise KeyboardInterrupt

    save_models(e, bleu=max_bleu, token_acc=max_token_acc, step=len(train_dataloader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--config", default=None, help="path to yaml config file", type=str
    )
    parser.add_argument("--no_cuda", action="store_true", help="Use CPU")
    parser.add_argument("--debug", action="store_true", help="DEBUG")
    parser.add_argument(
        "--resume", help="path to checkpoint folder", action="store_true"
    )
    parsed_args = parser.parse_args()
    if parsed_args.config is None:
        with in_model_path():
            parsed_args.config = os.path.realpath("settings/debug.yaml")

    params = read_yaml(parsed_args.config)
    args = parse_args(Munch(params), **vars(parsed_args))

    logging.getLogger().setLevel(
        logging.DEBUG if parsed_args.debug else logging.WARNING
    )
    seed_everything(args.seed)

    if args.wandb:
        if not parsed_args.resume:
            args.id = wandb.util.generate_id()
        wandb.init(config=dict(args), resume="allow", name=args.name, id=args.id)
        args = Munch(wandb.config)

    train(args)
