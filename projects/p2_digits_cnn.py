"""Project 2 — A convolutional classifier, trained by hand.

A complete PyTorch project on the bundled scikit-learn digits set (1,797
8x8 images, no download). Everything that a framework would hide is
written out:

    gradient check -> LR range test -> hand-written training loop ->
    determinism proof -> checkpoint round-trip -> confusion matrix ->
    calibration -> audit pack

The point is not the accuracy. The point is that every claim the run
makes is checked: the gradients against finite differences, the learning
rate against a measured curve, reproducibility against a second run, and
the saved model against the live one, bit for bit.

Chapters: 16 (autograd), 17 (activations, losses, init), 18 (the loop,
batching, optimisers, determinism), 19 (convolution).

Usage:
    python p2_digits_cnn.py
    python p2_digits_cnn.py --epochs 40 --seed 7
    python p2_digits_cnn.py --skip-lr-test
"""

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError as error:        # pragma: no cover - environment gate
    HAS_TORCH = False
    TORCH_ERROR = error

try:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError as error:        # pragma: no cover - environment gate
    HAS_SKLEARN = False
    SKLEARN_ERROR = error

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / 'runs'


# --------------------------------------------------------------------- #
#                         1. Data and model                              #
# --------------------------------------------------------------------- #


def load_data(seed: int) -> Tuple:
    """Loads the digits set and splits it three ways.

    A validation split exists so that the learning rate, the number of
    epochs and the calibration temperature are all chosen without ever
    touching the test set (Chapter 14).

    Args:
        seed: Seed for the splits.

    Returns:
        A tuple of six arrays: train/val/test features and labels.
    """
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0      # scale to [0, 1]
    labels = digits.target.astype(np.int64)

    x_dev, x_test, y_dev, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=seed, stratify=labels)
    x_train, x_val, y_train, y_val = train_test_split(
        x_dev, y_dev, test_size=0.2, random_state=seed, stratify=y_dev)
    return x_train, y_train, x_val, y_val, x_test, y_test


class SmallCNN(nn.Module):
    """A two-block convolutional classifier for 8x8 digits.

    Deliberately small enough to train on a CPU in seconds, and
    deliberately explicit: no Sequential magic hiding the shapes.

    Attributes:
        conv1: First convolution, 1 -> 16 channels.
        conv2: Second convolution, 16 -> 32 channels.
        head: The linear classifier on the pooled features.
    """

    def __init__(self, n_classes: int = 10) -> None:
        """Builds the layers.

        Args:
            n_classes: Number of output classes.
        """
        super().__init__()
        # 8x8 -> conv(pad 1) keeps 8x8 -> pool halves to 4x4
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # 4x4 -> conv(pad 1) keeps 4x4 -> pool halves to 2x2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(32 * 2 * 2, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Applies He initialisation, the right choice for ReLU.

        Var(w) = 2 / fan_in, the factor of two compensating for ReLU
        zeroing half its inputs (Chapter 17).
        """
        for module in (self.conv1, self.conv2, self.head):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Runs the forward pass.

        Args:
            x: Input batch of shape (batch, 1, 8, 8).

        Returns:
            Logits of shape (batch, n_classes). Logits, not
            probabilities: the loss fuses the softmax for stability.
        """
        x = self.pool(torch.relu(self.conv1(x)))       # (b, 16, 4, 4)
        x = self.pool(torch.relu(self.conv2(x)))       # (b, 32, 2, 2)
        x = self.dropout(x.flatten(1))
        return self.head(x)


def set_seed(seed: int) -> None:
    """Seeds every generator that affects a run.

    Args:
        seed: The seed.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int,
                shuffle: bool, seed: int = 0) -> 'DataLoader':
    """Wraps arrays in a deterministic DataLoader.

    Args:
        x: Images of shape (n, 8, 8).
        y: Integer labels.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        seed: Seed for the shuffling generator.

    Returns:
        A DataLoader yielding (images, labels).
    """
    tensors = TensorDataset(torch.tensor(x).unsqueeze(1), torch.tensor(y))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle,
                      generator=generator if shuffle else None,
                      num_workers=0)


# --------------------------------------------------------------------- #
#                       2. Checks before training                        #
# --------------------------------------------------------------------- #


def gradient_check(seed: int) -> float:
    """Checks autograd against central differences on the real model.

    Done in double precision on a handful of parameters: float32 round-off
    sits above the discrepancy you are hunting (Chapter 16).

    Args:
        seed: Seed for the probe batch.

    Returns:
        The largest relative discrepancy found.
    """
    set_seed(seed)
    model = SmallCNN().double()
    model.eval()                       # dropout off, or the check is noise
    x = torch.randn(4, 1, 8, 8, dtype=torch.float64)
    y = torch.randint(0, 10, (4,))
    loss_fn = nn.CrossEntropyLoss()

    loss_fn(model(x), y).backward()
    weight = model.conv1.weight
    analytic = weight.grad.clone()

    worst = 0.0
    h = 1e-6
    with torch.no_grad():
        for index in [(0, 0, 0, 0), (3, 0, 1, 2), (7, 0, 2, 1)]:
            original = weight[index].item()
            weight[index] = original + h
            up = float(loss_fn(model(x), y))
            weight[index] = original - h
            down = float(loss_fn(model(x), y))
            weight[index] = original
            numeric = (up - down) / (2 * h)
            scale = max(abs(numeric), 1e-8)
            worst = max(worst, abs(numeric - analytic[index].item()) / scale)
    return worst


def lr_range_test(x_train: np.ndarray, y_train: np.ndarray, seed: int,
                  low: float = 1e-4, high: float = 1.0,
                  steps: int = 60) -> Tuple[float, List[Tuple[float, float]]]:
    """Sweeps the learning rate geometrically and records the loss.

    The useful rate sits roughly an order of magnitude below the minimum
    of this curve -- the same stability edge derived in Chapter 6.

    Args:
        x_train: Training images.
        y_train: Training labels.
        seed: Seed.
        low: Starting learning rate.
        high: Final learning rate.
        steps: Number of batches to sweep over.

    Returns:
        A tuple (suggested_lr, curve) where curve is (lr, smoothed_loss).
    """
    set_seed(seed)
    model = SmallCNN()
    optimiser = torch.optim.SGD(model.parameters(), lr=low, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(x_train, y_train, 64, shuffle=True, seed=seed)

    multiplier = (high / low) ** (1.0 / max(steps - 1, 1))
    lr = low
    curve: List[Tuple[float, float]] = []
    smoothed = None
    model.train()
    iterator = iter(loader)
    for _ in range(steps):
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            xb, yb = next(iterator)
        for group in optimiser.param_groups:
            group['lr'] = lr
        optimiser.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimiser.step()
        value = float(loss)
        smoothed = value if smoothed is None else 0.8 * smoothed + 0.2 * value
        curve.append((lr, smoothed))
        if not math.isfinite(smoothed) or smoothed > 20.0:
            break
        lr *= multiplier

    best_lr = min(curve, key=lambda pair: pair[1])[0]
    return best_lr / 10.0, curve


# --------------------------------------------------------------------- #
#                           3. Training                                  #
# --------------------------------------------------------------------- #


def train_model(x_train, y_train, x_val, y_val, lr: float, epochs: int,
                batch_size: int, seed: int) -> Tuple['SmallCNN', List[Dict]]:
    """Trains with an explicit loop and returns the history.

    The four lines in the inner loop are the whole algorithm; everything
    else here is bookkeeping (Chapter 18).

    Args:
        x_train: Training images.
        y_train: Training labels.
        x_val: Validation images.
        y_val: Validation labels.
        lr: Peak learning rate.
        epochs: Number of epochs.
        batch_size: Batch size.
        seed: Seed.

    Returns:
        A tuple (model, history).
    """
    set_seed(seed)
    model = SmallCNN()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = make_loader(x_train, y_train, batch_size, True, seed)

    history: List[Dict] = []
    for epoch in range(epochs):
        model.train()
        running, batches = 0.0, 0
        for xb, yb in train_loader:
            optimiser.zero_grad()                 # 1. clear
            logits = model(xb)                    # 2. forward
            loss = loss_fn(logits, yb)
            loss.backward()                       # 3. backward
            optimiser.step()                      # 4. update
            # .detach() first: converting a tensor that still
            # carries requires_grad warns, and in a loop it keeps
            # the graph alive.
            running += float(loss.detach())
            batches += 1
        scheduler.step()

        val_logits = predict_logits(model, x_val)
        val_loss = float(loss_fn(val_logits, torch.tensor(y_val)))
        val_acc = float((val_logits.argmax(1).numpy() == y_val).mean())
        history.append({
            'epoch': epoch + 1,
            'train_loss': round(running / batches, 5),
            'val_loss': round(val_loss, 5),
            'val_accuracy': round(val_acc, 5),
            'lr': round(float(scheduler.get_last_lr()[0]), 6),
        })
    return model, history


def predict_logits(model: 'SmallCNN', x: np.ndarray) -> 'torch.Tensor':
    """Returns logits for an array of images, with the graph disabled.

    Args:
        model: The trained model.
        x: Images of shape (n, 8, 8).

    Returns:
        A tensor of logits, shape (n, 10).
    """
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x).unsqueeze(1))


# --------------------------------------------------------------------- #
#                         4. Evaluation                                  #
# --------------------------------------------------------------------- #


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int = 10) -> np.ndarray:
    """Builds a confusion matrix with truth on the rows.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        n_classes: Number of classes.

    Returns:
        An (n_classes, n_classes) integer array.
    """
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1
    return matrix


def macro_micro_f1(matrix: np.ndarray) -> Tuple[float, float]:
    """Computes macro and micro F1 from a confusion matrix.

    Both are returned because they answer different questions and can
    disagree sharply (Chapter 14).

    Args:
        matrix: Confusion matrix with truth on the rows.

    Returns:
        A tuple (macro_f1, micro_f1).
    """
    true_pos = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    actual = matrix.sum(axis=1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(predicted > 0, true_pos / predicted, 0.0)
        recall = np.where(actual > 0, true_pos / actual, 0.0)
        denom = precision + recall
        per_class = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    micro = float(true_pos.sum() / matrix.sum())
    return float(per_class.mean()), micro


def fit_temperature(logits: 'torch.Tensor', y: np.ndarray) -> float:
    """Fits a single temperature to calibrate the logits.

    Temperature scaling divides the logits by T > 0, which cannot change
    the ranking and therefore cannot change accuracy -- it only changes
    confidence. T > 1 means the model was over-confident.

    Args:
        logits: Validation logits.
        y: Validation labels.

    Returns:
        The fitted temperature.
    """
    log_t = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
    targets = torch.tensor(y)
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(logits / torch.exp(log_t), targets)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(torch.exp(log_t).item())


def expected_calibration_error(probabilities: np.ndarray, y: np.ndarray,
                               bins: int = 10) -> float:
    """Average gap between confidence and accuracy, weighted by bin size.

    Args:
        probabilities: Predicted class probabilities, shape (n, k).
        y: Ground-truth labels.
        bins: Number of confidence bins.

    Returns:
        The expected calibration error.
    """
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for i in range(bins):
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if mask.sum() == 0:
            continue
        error += (mask.mean() *
                  abs(float(correct[mask].mean() - confidence[mask].mean())))
    return float(error)


@dataclasses.dataclass
class Gate:
    """One acceptance check declared before the run.

    Attributes:
        name: Identifier.
        rule: Declared threshold, as text.
        observed: What the run produced.
        passed: Whether it held.
    """

    name: str
    rule: str
    observed: str
    passed: bool


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs the project end to end.

    Args:
        argv: Optional argument vector.

    Returns:
        0 when every gate passed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260101)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None,
                        help='override the LR range test')
    parser.add_argument('--skip-lr-test', action='store_true')
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args(argv)

    if not HAS_TORCH or not HAS_SKLEARN:
        missing = 'torch' if not HAS_TORCH else 'scikit-learn'
        print(f'UNVERIFIED: {missing} is not installed. '
              f'Run testbed/setup.ps1 first.')
        return 2

    started = time.perf_counter()
    print('=' * 74)
    print('PROJECT 2 - A convolutional classifier, trained by hand')
    print('=' * 74)

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.seed)
    print(f'\n[1] Data: train {len(x_train)}, val {len(x_val)}, '
          f'test {len(x_test)}; images {x_train.shape[1]}x{x_train.shape[2]}')

    # --- 2. gradient check before training anything ------------------- #
    worst = gradient_check(args.seed)
    print(f'\n[2] Gradient check (double precision, dropout off)')
    print(f'    worst relative gap autograd vs central difference: {worst:.2e}')

    params = sum(p.numel() for p in SmallCNN().parameters())
    print(f'    parameter count: {params:,}')

    # --- 3. learning rate ---------------------------------------------- #
    if args.lr is not None:
        lr, curve = args.lr, []
        print(f'\n[3] Learning rate supplied: {lr}')
    elif args.skip_lr_test:
        lr, curve = 0.05, []
        print(f'\n[3] LR range test skipped; using {lr}')
    else:
        lr, curve = lr_range_test(x_train, y_train, args.seed)
        diverged = [pair for pair in curve if not math.isfinite(pair[1])
                    or pair[1] > 10.0]
        print(f'\n[3] LR range test over {len(curve)} batches')
        print(f'    loss minimised near lr={min(curve, key=lambda p: p[1])[0]:.4f}')
        print(f'    chosen (one order below): {lr:.4f}')
        if diverged:
            print(f'    diverged from lr={diverged[0][0]:.3f} upward '
                  f'-- the 2/L edge of Chapter 6')

    # --- 4. train ------------------------------------------------------- #
    print(f'\n[4] Training {args.epochs} epochs, batch {args.batch_size}, '
          f'cosine schedule')
    model, history = train_model(x_train, y_train, x_val, y_val, lr,
                                 args.epochs, args.batch_size, args.seed)
    for row in history[::max(len(history) // 5, 1)]:
        print(f'    epoch {row["epoch"]:>3}  train {row["train_loss"]:.4f}  '
              f'val {row["val_loss"]:.4f}  acc {row["val_accuracy"]:.4f}  '
              f'lr {row["lr"]:.5f}')
    print(f'    final val accuracy {history[-1]["val_accuracy"]:.4f}')

    # --- 5. determinism -------------------------------------------------- #
    model_again, history_again = train_model(x_train, y_train, x_val, y_val,
                                             lr, args.epochs,
                                             args.batch_size, args.seed)
    identical = all(
        abs(a['train_loss'] - b['train_loss']) < 1e-12
        for a, b in zip(history, history_again))
    print(f'\n[5] Determinism: a second run at the same seed is '
          f'{"identical" if identical else "DIFFERENT"}')

    # --- 6. checkpoint round-trip ---------------------------------------- #
    checkpoint = {
        'model_state': model.state_dict(),
        'epochs': args.epochs,
        'lr': lr,
        'seed': args.seed,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / 'p2_digits_cnn' / 'checkpoint.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

    restored = SmallCNN()
    restored.load_state_dict(torch.load(path, weights_only=True)['model_state'])
    before = predict_logits(model, x_test)
    after = predict_logits(restored, x_test)
    max_diff = float((before - after).abs().max())
    print(f'\n[6] Checkpoint round-trip: max logit difference {max_diff:.2e}')

    # --- 7. test metrics -------------------------------------------------- #
    test_logits = predict_logits(model, x_test)
    predictions = test_logits.argmax(1).numpy()
    matrix = confusion_matrix(y_test, predictions)
    macro, micro = macro_micro_f1(matrix)
    accuracy = float((predictions == y_test).mean())
    print(f'\n[7] Test: accuracy {accuracy:.4f}  '
          f'macro-F1 {macro:.4f}  micro-F1 {micro:.4f}')
    confusions = [(int(i), int(j), int(matrix[i, j]))
                  for i in range(10) for j in range(10)
                  if i != j and matrix[i, j] > 0]
    confusions.sort(key=lambda t: -t[2])
    print('    most confused pairs: ' + ', '.join(
        f'{i}->{j} ({c})' for i, j, c in confusions[:5]) or '    none')

    # --- 8. calibration ---------------------------------------------------- #
    val_logits = predict_logits(model, x_val)
    temperature = fit_temperature(val_logits, y_val)

    val_probs_raw = torch.softmax(val_logits, dim=1).numpy()
    val_probs_cal = torch.softmax(val_logits / temperature, dim=1).numpy()
    val_ece_raw = expected_calibration_error(val_probs_raw, y_val)
    val_ece_cal = expected_calibration_error(val_probs_cal, y_val)

    raw_probs = torch.softmax(test_logits, dim=1).numpy()
    cal_probs = torch.softmax(test_logits / temperature, dim=1).numpy()
    ece_raw = expected_calibration_error(raw_probs, y_test)
    ece_cal = expected_calibration_error(cal_probs, y_test)
    cal_accuracy = float((cal_probs.argmax(1) == y_test).mean())

    # An ECE on n points is itself an estimate; its sampling noise is
    # roughly 1/sqrt(n), which here is about 0.05. Any test-set movement
    # smaller than that is not evidence of anything.
    noise = 1.0 / math.sqrt(len(y_test))

    print(f'\n[8] Temperature scaling: T = {temperature:.4f} '
          f'({"over" if temperature > 1 else "under"}-confident)')
    print(f'    on validation (where T was fitted): '
          f'ECE {val_ece_raw:.4f} -> {val_ece_cal:.4f}')
    print(f'    on test (held out):                 '
          f'ECE {ece_raw:.4f} -> {ece_cal:.4f}')
    print(f'    accuracy unchanged: {accuracy:.4f} -> {cal_accuracy:.4f}')
    if ece_raw < 0.05:
        print(f'    note: the model was already well calibrated '
              f'(ECE {ece_raw:.3f}), so there was little to win, and T is')
        print(f'    itself estimated from only {len(y_val)} points. '
              f'Test movement under +/-{noise:.3f} is sampling noise.')

    # --- 9. gates ----------------------------------------------------------- #
    gates = [
        Gate('gradient_check', 'autograd matches finite differences < 1e-5',
             f'{worst:.2e}', worst < 1e-5),
        Gate('determinism', 'same seed gives identical losses',
             'identical' if identical else 'different', identical),
        Gate('checkpoint_exact', 'reloaded model reproduces logits exactly',
             f'{max_diff:.2e}', max_diff == 0.0),
        Gate('beats_chance', 'accuracy > 0.90 on 10 classes',
             f'{accuracy:.4f}', accuracy > 0.90),
        Gate('calibration_improves_where_fitted',
             'ECE improves on the validation set T was fitted on',
             f'{val_ece_raw:.4f} -> {val_ece_cal:.4f}',
             val_ece_cal <= val_ece_raw + 1e-9),
        Gate('calibration_test_within_noise',
             f'test ECE moves less than 1/sqrt(n) = {noise:.3f}',
             f'{ece_raw:.4f} -> {ece_cal:.4f} '
             f'(delta {ece_cal - ece_raw:+.4f})',
             abs(ece_cal - ece_raw) < noise),
        Gate('temperature_preserves_accuracy',
             'temperature scaling cannot change the ranking',
             f'{accuracy:.4f} vs {cal_accuracy:.4f}',
             abs(accuracy - cal_accuracy) < 1e-12),
        Gate('loss_decreased', 'final train loss below the first epoch',
             f'{history[0]["train_loss"]:.4f} -> '
             f'{history[-1]["train_loss"]:.4f}',
             history[-1]['train_loss'] < history[0]['train_loss']),
    ]
    print(f'\n[9] Gates')
    for gate in gates:
        mark = 'PASS' if gate.passed else 'FAIL'
        print(f'    [{mark}] {gate.name}: {gate.observed}  '
              f'(rule: {gate.rule})')

    elapsed = time.perf_counter() - started
    status = 'VERIFIED' if all(g.passed for g in gates) else 'PENDING REVISION'
    print(f'\nstatus: {status}   ({elapsed:.1f}s)')

    if not args.no_write:
        run_dir = RUNS / 'p2_digits_cnn'
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'project': 'p2_digits_cnn',
            'status': status,
            'seed': args.seed,
            'parameters': params,
            'learning_rate': lr,
            'lr_curve': [[round(a, 6), round(b, 5)] for a, b in curve],
            'history': history,
            'gradient_check': worst,
            'test': {'accuracy': accuracy, 'macro_f1': macro,
                     'micro_f1': micro},
            'confusion_matrix': matrix.tolist(),
            'calibration': {'temperature': temperature,
                            'val_ece_before': val_ece_raw,
                            'val_ece_after': val_ece_cal,
                            'test_ece_before': ece_raw,
                            'test_ece_after': ece_cal,
                            'noise_floor': noise},
            'gates': [dataclasses.asdict(g) for g in gates],
            'torch_version': torch.__version__,
            'seconds': round(elapsed, 1),
        }
        (run_dir / 'report.json').write_text(
            json.dumps(payload, indent=2, default=str), encoding='utf-8')
        print(f'audit pack: {run_dir}')

    return 0 if all(g.passed for g in gates) else 1


if __name__ == '__main__':
    sys.exit(main())
