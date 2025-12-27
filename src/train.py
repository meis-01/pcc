import argparse, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import PCCDataset, PCCConfig
from .models import RealCNN, ComplexCNN, RealConstrainedComplexCNN
from .utils import set_seed, device, accuracy, save_json


def batch_to_inputs(batch, model_kind: str):
    z, A, theta, y = batch
    # z: complex (B,H,W) as complex64
    if model_kind == "mag":
        x = A.unsqueeze(1)  # (B,1,H,W)
        return (x,), y
    if model_kind in ["R2", "R2c"]:
        xr = z.real.unsqueeze(1)
        xi = z.imag.unsqueeze(1)
        x = torch.cat([xr, xi], dim=1)
        return (x,), y

    if model_kind == "cossin":
        x1 = torch.cos(theta).unsqueeze(1)
        x2 = torch.sin(theta).unsqueeze(1)
        xA = A.unsqueeze(1)
        x = torch.cat([x1, x2, xA], dim=1)  # 3-ch real input
        return (x,), y
    if model_kind == "complex":
        xr = z.real.unsqueeze(1)
        xi = z.imag.unsqueeze(1)
        return (xr, xi), y
    raise ValueError(f"Unknown model {model_kind}")


def make_model(model_kind: str):
    if model_kind == "mag":
        return RealCNN(in_ch=1, width=32)
    if model_kind == "R2":
        return RealCNN(in_ch=2, width=32)
    if model_kind == "R2c":  # <- NEW constrained-real model
        return RealConstrainedComplexCNN(in_ch_complex=1, width=24)
    if model_kind == "cossin":
        return RealCNN(in_ch=3, width=32)
    if model_kind == "complex":
        return ComplexCNN(in_ch=1, width=24)
    raise ValueError(model_kind)


@torch.no_grad()
def eval_loader(model, loader, model_kind: str, dev):
    model.eval()
    total_acc, total_n, total_loss = 0.0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    for batch in loader:
        batch = [b.to(dev) if torch.is_tensor(b) else b for b in batch]
        inputs, y = batch_to_inputs(batch, model_kind)
        logits = model(*inputs) if model_kind == "complex" else model(*inputs)
        loss = ce(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += (torch.argmax(logits, dim=1) == y).float().sum().item()
        total_n += bs
    return total_loss / total_n, total_acc / total_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", choices=["mag", "R2", "R2c", "cossin", "complex"], default="complex"
    )
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--train_size", type=int, default=1000)
    ap.add_argument("--val_size", type=int, default=200)
    ap.add_argument("--test_size", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_dir", type=str, default="runs")
    args = ap.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    set_seed(args.seed)
    dev = device()

    cfg = PCCConfig(N=args.N)
    train_ds = PCCDataset(size=args.train_size, cfg=cfg, seed=args.seed + 1)
    val_ds = PCCDataset(size=args.val_size, cfg=cfg, seed=args.seed + 2)
    test_ds = PCCDataset(size=args.test_size, cfg=cfg, seed=args.seed + 3)
    print(
        f"Dataset sizes | train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}"
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"Using model: {args.model}")
    model = make_model(args.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    ce = nn.CrossEntropyLoss()
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(
        args.run_dir, f"pcc_{args.model}_N{args.N}_seed{args.seed}_{timestamp}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")
    best_val = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    history = []
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        print("-" * 60)
        print(f"Epoch {epoch}/{args.epochs}")
        model.train()
        t0 = time.time()
        running_loss, running_acc, n = 0.0, 0.0, 0
        for ix, batch in enumerate(train_loader):
            batch = [b.to(dev) if torch.is_tensor(b) else b for b in batch]
            inputs, y = batch_to_inputs(batch, args.model)

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
        va_loss, va_acc = eval_loader(model, val_loader, args.model, dev)

        if va_acc > best_val:
            best_val = va_acc
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "cfg": cfg.__dict__},
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
        print(
            f"Epoch {epoch:02d} | train acc {tr_acc:.3f} loss {tr_loss:.3f} | val acc {va_acc:.3f} loss {va_loss:.3f}"
        )

    # final test using best checkpoint
    ckpt = torch.load(best_path, map_location=dev)
    model.load_state_dict(ckpt["model"])
    te_loss, te_acc = eval_loader(model, test_loader, args.model, dev)

    summary = {
        "task": "Phase-Coherence Classification (PCC)",
        "model": args.model,
        "N": args.N,
        "seed": args.seed,
        "best_val_acc": best_val,
        "test_acc": te_acc,
        "test_loss": te_loss,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "cfg": cfg.__dict__,
        "history": history,
    }
    save_json(os.path.join(out_dir, "summary.json"), summary)
    print(f"\nBest val acc: {best_val:.4f}")
    print(f"Test acc:     {te_acc:.4f}")
    print(f"Saved to:     {out_dir}")


if __name__ == "__main__":
    main()
