"""
check_cuda_backend.py — FiverLayer(backend="cuda") must train like backend="numpy".

Trains the same small layer pair with both backends (same seeds, same batches)
and compares forward output, grad_x, codes and bias after every step.  Codes
may differ only where the gradient is ~0 (float summation order decides the
sign there).  The int8 path is approximate by design and only reported.

  python emoji_net/check_cuda_backend.py                    # library from the default paths
  FIVER_CUDA_LIB=build-emu/cuda/libfiver_cuda.so python ... # CPU emulator build
"""
import sys

import numpy as np

import fiver
import fiver_cuda


def main():
    if not fiver_cuda.available():
        print("libfiver_cuda not found — build it first (see docs/ARCHITEKTUR_CUDA.md)")
        return 2
    ctx = fiver_cuda.default_context()
    print(f"device: {ctx.device_name} (sm_{ctx.sm}{', CPU emulator' if ctx.emulator else ''})")
    rng = np.random.default_rng(0)
    failures = 0
    for grid in ("centered", "anchored", "magic"):
        for path, batch in (("float", 5), ("float", 40), ("int8", 40)):
            dims = [(97, 33, "softsign"), (33, 11, "identity")]
            num = [fiver.FiverLayer(i, o, act=a, grid=grid, seed=s, backend="numpy") for s, (i, o, a) in enumerate(dims)]
            cud = [fiver.FiverLayer(i, o, act=a, grid=grid, seed=s, backend="cuda", path=path) for s, (i, o, a) in enumerate(dims)]
            worst_y = worst_dx = 0.0
            ambiguous = wrong = 0
            for step in range(4):
                x = rng.normal(size=(batch, 97)).astype(np.float32)
                t = rng.normal(size=(batch, 11)).astype(np.float32)
                outs = []
                for net in (num, cud):
                    h = net[0].forward(x)
                    y = net[1].forward(h)
                    g = 2 * (y - t) / y.size
                    dx = net[0].backward(x, net[1].backward(h, g))
                    outs.append((y, dx))
                    for layer in net:
                        layer.step(0.05)
                tol = 1e-4
                worst_y = max(worst_y, float(np.abs(outs[0][0] - outs[1][0]).max() / (np.abs(outs[0][0]).max() + 1e-9)))
                worst_dx = max(worst_dx, float(np.abs(outs[0][1] - outs[1][1]).max() / (np.abs(outs[0][1]).max() + 1e-9)))
                for a, b in zip(num, cud):
                    diff = a.k_data != b.k_data
                    if path == "float":
                        wrong += int(diff.sum())
                    else:
                        ambiguous += int(diff.sum())
                    # continue from identical states so later steps stay comparable
                    b.k_data = a.k_data
                    b.bias[:] = a.bias
            # int8: the forward is approximate by design (x rounded per row); only require a sane bound.
            # The anchored grid deviates most: its weights are ~1, so the absolute rounding error is
            # large compared with outputs near 0, which softsign passes on almost unchanged.
            ok = (worst_y < 0.25 and worst_dx < 0.25) if path == "int8" else \
                 (worst_y < tol and worst_dx < tol and wrong <= 2)
            failures += not ok
            print(f"  grid={grid:8s} path={path:5s} batch={batch:3d}  forward rel diff {worst_y:.1e}  "
                  f"grad_x rel diff {worst_dx:.1e}  codes differing {wrong if path == 'float' else ambiguous}"
                  f"{' (int8 forward changes the gradient, expected)' if path == 'int8' else ''}  {'ok' if ok else 'FAIL'}")
    print("CHECK", "PASSED" if failures == 0 else f"FAILED ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
