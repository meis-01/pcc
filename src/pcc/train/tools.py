import torch
import torch.nn as nn


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


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()
