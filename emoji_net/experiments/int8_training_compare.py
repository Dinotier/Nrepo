"""
int8_training_compare.py — does STE training stay as good with the INT8 forward path?

The INT8 path of libfiver_cuda rounds x per input row to int8 (sx = max|x|/127,
round half to even) and uses the int8 weight codes e (v ≈ offset + e/E).  This
script reproduces that forward exactly in numpy, keeps the backward in float
(as on the GPU), and trains the same network twice per seed — once with the
float forward, once with the INT8 forward — on identical data, initialisation
and batch order (paired comparison).

Task: teacher–student regression.  A random float teacher MLP labels Gaussian
inputs; the student is a two-layer Fiver network (softsign hidden layer, as in
EmojiNet).  Many batches and several seeds average out batch-to-batch noise.

Reported: held-out MSE (always evaluated with the exact float forward) as mean
± std over seeds, the paired difference int8 − float, and the curves.

  python emoji_net/experiments/int8_training_compare.py --seeds 8 --steps 3000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fiver import FiverLayer, _apply_act  # noqa: E402

E_TABLE = {    # identical to FIVER_E_INIT in cuda/include/fiver/format.h
    "centered": [-127, -107, -90, -76, -64, -53, -45, -38, -32, -27, -22, -19, -16, -13, -11, -9,
                 0, 11, 13, 16, 19, 22, 27, 32, 38, 45, 53, 64, 76, 90, 107, 127],
    "magic":    [-64, -56, -48, -40, -32, -28, -24, -20, -16, -14, -12, -10, -8, -7, -6, -5,
                 0, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64],
}
E_SCALE = {"centered": 127.0, "magic": 64.0}


class Int8Layer(FiverLayer):
    """FiverLayer whose forward pre-activation follows the GPU INT8 path exactly."""

    def _pre_act(self, x):
        x2 = x.reshape(-1, self.in_dim).astype(np.float32)
        m = np.abs(x2).max(axis=1, keepdims=True)
        inv = np.where(m > 0, np.float32(127.0) / np.where(m > 0, m, 1), 0).astype(np.float32)
        xq = np.clip(np.rint(x2 * inv), -127, 127).astype(np.float64)
        sx = (m / np.float32(127.0)).astype(np.float64)
        e = np.asarray(E_TABLE[self.grid], dtype=np.float64)[self.k_data]
        acc = xq @ e.T                                            # exact: integers < 2^53
        y = acc * (float(self.scale) / E_SCALE[self.grid]) * sx + self.bias
        return y.astype(np.float32).reshape(*x.shape[:-1], self.out_dim)


def make_task(seed, n_in, n_hid, n_out, n_train, n_test):
    rng = np.random.default_rng(1000 + seed)
    w1 = rng.normal(size=(n_hid, n_in)).astype(np.float32) / np.sqrt(n_in)
    w2 = rng.normal(size=(n_out, n_hid)).astype(np.float32) / np.sqrt(n_hid)
    teacher = lambda x: np.tanh(x @ w1.T) @ w2.T
    xtr = rng.normal(size=(n_train, n_in)).astype(np.float32)
    xte = rng.normal(size=(n_test, n_in)).astype(np.float32)
    return xtr, teacher(xtr).astype(np.float32), xte, teacher(xte).astype(np.float32)


class Student:
    def __init__(self, cls, n_in, n_hid, n_out, grid, seed):
        self.l1 = cls(n_in, n_hid, act="softsign", grid=grid, seed=2 * seed + 1, backend="numpy")
        self.l2 = cls(n_hid, n_out, act="identity", grid=grid, seed=2 * seed + 2, backend="numpy")

    def forward(self, x):
        h = self.l1.forward(x)
        return h, self.l2.forward(h)

    def train_step(self, x, y, lr):
        h, out = self.forward(x)
        g = 2.0 * (out - y) / out.size
        self.l1.backward(x, self.l2.backward(h, g))
        self.l1.step(lr)
        self.l2.step(lr)
        return float(((out - y) ** 2).mean())

    def float_eval(self, x, y):
        h = _apply_act(x @ self.l1.materialize().T + self.l1.bias, self.l1.act)
        out = h @ self.l2.materialize().T + self.l2.bias
        return float(((out - y) ** 2).mean())


def run(mode, grid, seed, task, args):
    xtr, ytr, xte, yte = task
    s = Student(Int8Layer if mode == "int8" else FiverLayer, args.n_in, args.n_hid, args.n_out, grid, seed)
    order = np.random.default_rng(seed).integers(0, len(xtr), size=(args.steps, args.batch))
    curve, train = [], []
    for step in range(args.steps):
        idx = order[step]
        train.append(s.train_step(xtr[idx], ytr[idx], args.lr))
        if (step + 1) % args.eval_every == 0:
            curve.append(s.float_eval(xte, yte))
    return np.array(curve), float(np.mean(train[-args.eval_every:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--n-in", type=int, default=512)
    ap.add_argument("--n-hid", type=int, default=256)
    ap.add_argument("--n-out", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--grids", default="centered,magic")
    args = ap.parse_args()

    t0 = time.time()
    print(f"teacher-student {args.n_in}->{args.n_hid}->{args.n_out}, batch {args.batch}, "
          f"{args.steps} steps, {args.seeds} seeds, held-out {args.n_test}\n")
    for grid in args.grids.split(","):
        curves = {"float": [], "int8": []}
        finals = {"float": [], "int8": []}
        for seed in range(args.seeds):
            task = make_task(seed, args.n_in, args.n_hid, args.n_out, args.n_train, args.n_test)
            for mode in ("float", "int8"):
                c, _ = run(mode, grid, seed, task, args)
                curves[mode].append(c)
                finals[mode].append(c[-1])
        cf, ci = np.array(curves["float"]), np.array(curves["int8"])
        ff, fi = np.array(finals["float"]), np.array(finals["int8"])
        d = fi - ff
        print(f"grid = {grid}")
        print(f"  {'step':>6}  {'float MSE (mean ± std)':>24}  {'int8 MSE (mean ± std)':>24}  {'int8 − float (paired)':>24}")
        for j in range(cf.shape[1]):
            dd = ci[:, j] - cf[:, j]
            print(f"  {(j + 1) * args.eval_every:>6}  {cf[:, j].mean():>12.5f} ± {cf[:, j].std():<9.5f}"
                  f"  {ci[:, j].mean():>12.5f} ± {ci[:, j].std():<9.5f}  {dd.mean():>+12.5f} ± {dd.std():<9.5f}")
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        rel = d.mean() / ff.mean()
        print(f"  final: float {ff.mean():.5f} ± {ff.std():.5f}   int8 {fi.mean():.5f} ± {fi.std():.5f}   "
              f"paired diff {d.mean():+.5f} ± {se:.5f} (s.e.)  = {100 * rel:+.2f} % of float;  "
              f"seed-to-seed std of float {100 * ff.std() / ff.mean():.2f} %\n")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
