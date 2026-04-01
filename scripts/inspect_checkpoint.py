import sys
import os
import torch

if len(sys.argv) < 2:
    print('usage: python scripts/inspect_checkpoint.py <checkpoint.pth>')
    sys.exit(1)

path = sys.argv[1]
if not os.path.exists(path):
    print('file not found:', path)
    sys.exit(2)

print('Path:', path)
print('Size (bytes):', os.path.getsize(path))
ckpt = torch.load(path, map_location='cpu')
print('Top-level keys:', list(ckpt.keys()))

if 'model_state' in ckpt:
    ms = ckpt['model_state']
    total = 0
    print('\nModel state tensors:')
    for name, tensor in ms.items():
        shape = tuple(tensor.shape) if hasattr(tensor, 'shape') else None
        print(f' - {name}: {shape}')
        try:
            total += tensor.numel()
        except Exception:
            pass
    print('Total model params:', total)

if 'optimizer' in ckpt:
    print('\nOptimizer keys:', list(ckpt['optimizer'].keys()))

if 'epoch' in ckpt:
    print('\nSaved epoch:', ckpt['epoch'])

# print a small sample of a parameter tensor (first param)
try:
    first = next(iter(ms.items()))
    name, tensor = first
    flat = tensor.view(-1)
    print('\nSample values from first parameter (%s):' % name)
    print(flat[:10].tolist())
except Exception:
    pass
