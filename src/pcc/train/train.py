import argparse, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pcc.dataset.dataset import PCCDataset
from pcc.module.net import make_model
from pcc.train.tools import batch_to_inputs, eval_loader, accuracy
from pcc.utils import device, save_json
from pcc.params import TrainConfig


def train(config=TrainConfig):

    train_ds = PCCDataset(
        size=config.hyperparams.train_size,
        cfg=config.pcc,
        seed=config.hyperparams.seed + 1,
    )
    val_ds = PCCDataset(
        size=config.hyperparams.val_size,
        cfg=config.pcc,
        seed=config.hyperparams.seed + 2,
    )
    test_ds = PCCDataset(
        size=config.hyperparams.test_size,
        cfg=config.pcc,
        seed=config.hyperparams.seed + 3,
    )
    print(
        f"Dataset sizes | train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}"
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.hyperparams.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.hyperparams.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.hyperparams.batch_size, shuffle=False, num_workers=0
    )
    print(f"Using model: {config.model}")
    model = make_model(config.model, config.normalize).to(device())
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.hyperparams.lr, weight_decay=1e-2
    )
    ce = nn.CrossEntropyLoss()
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )
    pcc_dict = {
        k: v
        for k, v in config.pcc.__dict__.items()
        if not k.startswith("__") and not callable(v)
    }
    hyp_dict = {
        k: v
        for k, v in config.hyperparams.__dict__.items()
        if not k.startswith("__") and not callable(v)
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(
        config.hyperparams.run_dir,
        f"pcc_{config.model}_N{config.pcc.N}_seed{config.hyperparams.seed}_{timestamp}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")
    best_val = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    history = []
    print("Starting training...")
    for epoch in range(1, config.hyperparams.epochs + 1):
        print("-" * 60)
        print(f"Epoch {epoch}/{config.hyperparams.epochs}")
        model.train()
        t0 = time.time()
        running_loss, running_acc, n = 0.0, 0.0, 0
        for ix, batch in enumerate(train_loader):
            batch = [b.to(device()) if torch.is_tensor(b) else b for b in batch]
            inputs, y = batch_to_inputs(batch, config.model)

            opt.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            bs = y.size(0)
            running_loss += loss.item() * bs
            running_acc += (torch.argmax(logits, dim=1) == y).float().sum().item()
            n += bs
            if torch.isnan(loss):
                ending = "\n"
            else:
                ending = "\r"
            print(
                f"  Batch {ix+1}/{len(train_loader)} -- loss: {loss.item():.4f} running_loss: {running_loss:.4f} n {n} normed_loss: {running_loss/n:.4f}",
                end=ending,
            )
        tr_loss = running_loss / n
        tr_acc = running_acc / n
        va_loss, va_acc = eval_loader(model, val_loader, config.model, device())

        if va_acc > best_val:
            best_val = va_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "pcc": pcc_dict,
                    "normalize": config.normalize,
                    "model_kind": config.model,
                    "hyperparams": hyp_dict,
                    "epoch": epoch,
                },
                best_path,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "sec": time.time() - t0,
            }
        )
        # Flush the line to clear any leftover '\r' output
        print(" " * 80, end="\r", flush=True)
        print(f"Model used: {config.model} | Normalize: {config.normalize}")
        print(
            f"Epoch {epoch:02d} | train acc {tr_acc:.3f} loss {tr_loss:.3f} | val acc {va_acc:.3f} loss {va_loss:.3f}"
        )

    # final test using best checkpoint
    ckpt = torch.load(best_path, map_location=device())
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = eval_loader(model, test_loader, config.model, device())

    summary = {
        "task": "Phase-Coherence Classification (PCC)",
        "model": config.model,
        "N": config.N,
        "seed": config.hyperparams.seed,
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_size": config.hyperparams.train_size,
        "val_size": config.hyperparams.val_size,
        "test_size": config.hyperparams.test_size,
        "pcc": config.pcc.__dict__,
        "history": history,
    }
    save_json(os.path.join(out_dir, "summary.json"), summary)
    print(f"\nBest val acc: {best_val:.4f}")
    print(f"Test acc:     {test_acc:.4f}")
    print(f"Saved to:     {out_dir}")


if __name__ == "__main__":
    train()
