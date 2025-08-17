# PainFormer

A Vision Foundation Model for Affective Computing and Automatic Pain Assessment

> **PainFormer v1.0** · **160-D embeddings** · **PyTorch ≥ 2.0**

---

## Paper

[**PainFormer: A Vision Foundation Model for Automatic Pain Assessment**](https://arxiv.org/abs/XXXX.XXXXX)

---

## Highlights

| Feature                | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| **Pre‑training scale** | Multi‑task pre‑training on **14 tasks / 10.9 M samples**. |
| **Parameters**         | **19.60 M** (PainFormer encoder).                         |
| **Compute**            | **5.82 GFLOPs** at 224×224 input.                         |
| **Embeddings**         | Fixed **160‑D** output vectors.                           |

<br/>

<p align="center">
  <img src="docs/painformer_overview.png" alt="PainFormer overview" width="65%"/>
</p>

<p align="center"><b>Figure&nbsp;1.</b> PainFormer overview (placeholder).</p>

---

## Table of Contents

1. [Pre-trained checkpoint](#pre-trained-checkpoint)
2. [Quick start](#quick-start)

   * [Extract embeddings](#extract-embeddings)
3. [Fine-tuning](#fine-tuning)
4. [Citation](#citation)
5. [Licence & acknowledgements](#licence--acknowledgements)
6. [Contact](#contact)

---

## Pre-trained checkpoint

Get the weights from the **[GitHub Releases](https://github.com/your-org/PainFormer/releases)** (placeholder).

| File             | Size    |
| ---------------- | ------- |
| `painformer.pth` | **TBA** |

```bash
# download the latest checkpoint (placeholder)
auto=https://github.com/your-org/PainFormer/releases/latest/download/painformer.pth
curl -L -o painformer.pth "$auto"

# optional: verify
sha256sum painformer.pth
```

The checkpoint contains **one key**:

```text
model_state_dict    # PainFormer backbone weights
```

---

## Quick start

> Assumes **PyTorch ≥ 2.0** and **timm ≥ 0.9** are installed.

### Extract embeddings

```python
import torch
from timm.models import create_model
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------
# Setup ---------------------------------------------------------
# ---------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG-Face2 stats used in our experiments
normalize = transforms.Normalize(
    mean=[0.6068, 0.4517, 0.3800],
    std=[0.2492, 0.2173, 0.2082]
)
to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# ---------------------------------------------------------------
# Load PainFormer -----------------------------------------------
# ---------------------------------------------------------------
model = create_model('painformer').to(device)
state = torch.load('./checkpoints/painformer.pth', map_location='cpu')  # expects 'model_state_dict'
model.load_state_dict(state['model_state_dict'], strict=False)

# expose embeddings (remove classification head)
model.head = torch.nn.Identity()
model.eval()

# ---------------------------------------------------------------
# One image → 160-D embedding -----------------------------------
# ---------------------------------------------------------------
img = Image.open('frame.png').convert('RGB')
x = to_tensor(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

with torch.no_grad():
    emb = model(x)        # [1, 160]
    emb = emb.squeeze(0)  # [160]

print("Embedding shape:", tuple(emb.shape))  # (160)
```

---

## Fine-tuning

```python
import torch, torch.nn as nn
from timm.models import create_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------------
# Config ---------------------------------------------------------
# ---------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 3   # set to your task

normalize = transforms.Normalize(
    mean=[0.6068, 0.4517, 0.3800],
    std=[0.2492, 0.2173, 0.2082]
)
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

train_set = datasets.ImageFolder('/path/to/train', transform=train_tf)
val_set   = datasets.ImageFolder('/path/to/val',   transform=val_tf)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

# ---------------------------------------------------------------
# Model ----------------------------------------------------------
# ---------------------------------------------------------------
model = create_model('painformer').to(device)
state = torch.load('./checkpoints/painformer.pth', map_location='cpu')
model.load_state_dict(state['model_state_dict'], strict=False)

# replace classifier to match downstream labels (PainFormer → 160-D)
model.head = nn.Linear(160, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# ---------------------------------------------------------------
# Train / Validate ----------------------------------------------
# ---------------------------------------------------------------
best = 0.0
for epoch in range(50):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
    acc = correct / max(total, 1)

    # save best
    if acc > best:
        best = acc
        torch.save({'model_state_dict': model.state_dict()}, './checkpoints/painformer_finetuned.pth')
    print(f"epoch {epoch:02d} | val acc {acc:.4f}")
```

---

## Citation

```bibtex
@article{gkikas2025painformer,
  title   = {PainFormer: A Vision Foundation Model for Automatic Pain Assessment},
  author  = {Gkikas, Stefanos and Rojas, Raul Fernandez and Tsiknakis, Manolis},
  journal = {to appear},
  year    = {2025}
}
```

---

## Licence & acknowledgements

* Code & weights: **MIT Licence** (placeholder) – see `LICENSE`

---

## Contact

Email **Stefanos Gkikas** ([gkikas@ics.forth.gr](mailto:gkikas@ics.forth.gr)).
