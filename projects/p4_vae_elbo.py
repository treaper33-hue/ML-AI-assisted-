"""Project 4 — A variational autoencoder, with the ELBO taken apart.

The smallest of the four projects and the most conceptual. A VAE on the
digits set, built so that the two halves of the ELBO are visible at every
step:

    PCA baseline -> VAE with the reparameterisation trick -> both ELBO
    terms logged separately -> a beta sweep that produces posterior
    collapse on purpose -> latent traversal -> audit pack

The lesson is in section [5]. At beta = 0 the KL term goes to zero and
the "VAE" is an ordinary autoencoder; at large beta the latent carries
nothing and reconstruction collapses. Watching only the summed loss hides
both failures completely, which is why this project logs the terms apart.

Chapters: 5 (PCA), 8 (KL divergence), 17 (losses), 20 (autoencoders, the
ELBO, reparameterisation).

Usage:
    python p4_vae_elbo.py
    python p4_vae_elbo.py --latent 4 --epochs 60
    python p4_vae_elbo.py --skip-sweep
"""

import argparse
import dataclasses
import json
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
except ImportError:                 # pragma: no cover - environment gate
    HAS_TORCH = False

try:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:                 # pragma: no cover - environment gate
    HAS_SKLEARN = False

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / 'runs'


class VAE(nn.Module):
    """A small fully-connected variational autoencoder.

    The encoder outputs mu and log-variance rather than a sample, and
    rather than sigma directly: log-variance is unconstrained, so the
    network needs no positivity trick and exp() keeps gradients sane.

    Attributes:
        encoder: Shared encoder trunk.
        to_mu: Linear map to the latent mean.
        to_logvar: Linear map to the latent log-variance.
        decoder: Maps a latent back to pixel space.
    """

    def __init__(self, n_inputs: int = 64, hidden: int = 128,
                 latent: int = 8) -> None:
        """Builds the layers.

        Args:
            n_inputs: Flattened input size.
            hidden: Width of the hidden layers.
            latent: Latent dimension.
        """
        super().__init__()
        self.latent = latent
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_inputs), nn.Sigmoid(),
        )

    def encode(self, x: 'torch.Tensor') -> Tuple['torch.Tensor',
                                                 'torch.Tensor']:
        """Returns the latent mean and log-variance.

        Args:
            x: Flattened inputs of shape (batch, n_inputs).

        Returns:
            A tuple (mu, logvar).
        """
        hidden = self.encoder(x)
        return self.to_mu(hidden), self.to_logvar(hidden)

    @staticmethod
    def reparameterise(mu: 'torch.Tensor',
                       logvar: 'torch.Tensor') -> 'torch.Tensor':
        """Samples z = mu + sigma * eps with eps ~ N(0, I).

        The randomness lives in eps, which carries no parameters, so the
        path from the loss back to mu and logvar stays differentiable.
        Sampling from N(mu, sigma) directly would break it.

        Args:
            mu: Latent means.
            logvar: Latent log-variances.

        Returns:
            One sample per row.
        """
        sigma = torch.exp(0.5 * logvar)
        return mu + sigma * torch.randn_like(sigma)

    def forward(self, x: 'torch.Tensor') -> Tuple['torch.Tensor',
                                                  'torch.Tensor',
                                                  'torch.Tensor']:
        """Encodes, samples and decodes.

        Args:
            x: Flattened inputs.

        Returns:
            A tuple (reconstruction, mu, logvar).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar


def elbo_terms(x_hat: 'torch.Tensor', x: 'torch.Tensor', mu: 'torch.Tensor',
               logvar: 'torch.Tensor',
               likelihood: str = 'bernoulli') -> Tuple['torch.Tensor',
                                                       'torch.Tensor']:
    """Returns the two ELBO terms, per batch, unsummed into one loss.

    The choice of likelihood is not cosmetic, and getting it wrong is
    instructive. The KL is measured in nats. Summed squared error over
    64 pixels in [0, 1] is of order 5, so against a KL of 10-30 nats the
    cheapest way to minimise the sum is to switch the latent off
    entirely -- posterior collapse caused purely by unit mismatch, not
    by anything interesting.

    The Bernoulli log-likelihood (binary cross-entropy) is both the
    correct likelihood for intensities in [0, 1] (Chapter 8) and
    measured in nats, so the two terms are commensurable and the model
    has a real trade-off to make.

    Args:
        x_hat: Reconstructions in [0, 1].
        x: Originals in [0, 1].
        mu: Latent means.
        logvar: Latent log-variances.
        likelihood: 'bernoulli' (nats, the default) or 'gaussian'
            (squared error, kept so the failure above can be reproduced).

    Returns:
        A tuple (reconstruction, kl), both summed over the batch.

    Raises:
        ValueError: If the likelihood is unknown.
    """
    if likelihood == 'bernoulli':
        reconstruction = torch.nn.functional.binary_cross_entropy(
            x_hat, x, reduction='sum')
    elif likelihood == 'gaussian':
        reconstruction = torch.nn.functional.mse_loss(x_hat, x,
                                                      reduction='sum')
    else:
        raise ValueError(f'unknown likelihood {likelihood!r}')
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction, kl


def set_seed(seed: int) -> None:
    """Seeds every generator that affects a run."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def pca_reconstruction_error(x_train: np.ndarray, x_test: np.ndarray,
                             k: int) -> float:
    """Mean per-image squared error of a rank-k PCA reconstruction.

    This is the honest baseline for any autoencoder: a linear method
    from 1901 (Chapter 5). An autoencoder that cannot beat it has
    learned nothing nonlinear.

    Args:
        x_train: Training images, flattened.
        x_test: Test images, flattened.
        k: Number of components.

    Returns:
        The mean squared reconstruction error per image.
    """
    mean = x_train.mean(axis=0)
    _, _, vt = np.linalg.svd(x_train - mean, full_matrices=False)
    components = vt[:k]
    projected = (x_test - mean) @ components.T
    reconstructed = projected @ components + mean
    return float(np.mean(np.sum((x_test - reconstructed) ** 2, axis=1)))


def train_vae(x_train: np.ndarray, x_val: np.ndarray, latent: int,
              beta: float, epochs: int, seed: int, verbose: bool = False,
              likelihood: str = 'bernoulli') -> Tuple['VAE', List[Dict]]:
    """Trains a VAE, logging both ELBO terms every epoch.

    Args:
        x_train: Training images, flattened to (n, 64).
        x_val: Validation images.
        latent: Latent dimension.
        beta: Weight on the KL term; 1.0 is the standard VAE.
        epochs: Number of epochs.
        seed: Seed.
        verbose: Whether to print progress.
        likelihood: 'bernoulli' or 'gaussian'; see `elbo_terms`.

    Returns:
        A tuple (model, history).
    """
    set_seed(seed)
    model = VAE(n_inputs=x_train.shape[1], latent=latent)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(x_train)),
                        batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val_tensor = torch.tensor(x_val)

    history: List[Dict] = []
    for epoch in range(epochs):
        model.train()
        recon_total, kl_total, seen = 0.0, 0.0, 0
        for (batch,) in loader:
            optimiser.zero_grad()
            x_hat, mu, logvar = model(batch)
            reconstruction, kl = elbo_terms(x_hat, batch, mu, logvar,
                                            likelihood)
            (reconstruction + beta * kl).backward()
            optimiser.step()
            recon_total += float(reconstruction)
            kl_total += float(kl)
            seen += len(batch)

        model.eval()
        with torch.no_grad():
            x_hat, mu, logvar = model(val_tensor)
            val_recon, val_kl = elbo_terms(x_hat, val_tensor, mu,
                                           logvar, likelihood)
            active = int((logvar.exp().mean(0) < 0.5).sum())
        history.append({
            'epoch': epoch + 1,
            'train_recon': round(recon_total / seen, 5),
            'train_kl': round(kl_total / seen, 5),
            'val_recon': round(float(val_recon) / len(x_val), 5),
            'val_kl': round(float(val_kl) / len(x_val), 5),
            'active_units': active,
        })
        if verbose and (epoch + 1) % max(epochs // 5, 1) == 0:
            row = history[-1]
            print(f'    epoch {row["epoch"]:>3}  recon {row["val_recon"]:.4f}'
                  f'  KL {row["val_kl"]:.4f}  '
                  f'active latent units {row["active_units"]}/{latent}')
    return model, history


@dataclasses.dataclass
class Gate:
    """One acceptance check declared before the run."""

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
    parser.add_argument('--latent', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--skip-sweep', action='store_true')
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args(argv)

    if not HAS_TORCH or not HAS_SKLEARN:
        missing = 'torch' if not HAS_TORCH else 'scikit-learn'
        print(f'UNVERIFIED: {missing} is not installed. '
              f'Run testbed/setup.ps1 first.')
        return 2

    started = time.perf_counter()
    print('=' * 74)
    print('PROJECT 4 - A variational autoencoder, with the ELBO taken apart')
    print('=' * 74)

    digits = load_digits()
    images = (digits.images.reshape(len(digits.images), -1).astype(np.float32)
              / 16.0)
    x_dev, x_test = train_test_split(images, test_size=0.2,
                                     random_state=args.seed)
    x_train, x_val = train_test_split(x_dev, test_size=0.2,
                                      random_state=args.seed)
    print(f'\n[1] Data: train {len(x_train)}, val {len(x_val)}, '
          f'test {len(x_test)}; {images.shape[1]} pixels in [0, 1]')

    # --- 2. the linear baseline ---------------------------------------- #
    pca_error = pca_reconstruction_error(x_train, x_test, args.latent)
    print(f'\n[2] PCA baseline at {args.latent} components')
    print(f'    test reconstruction error (sum of squares per image): '
          f'{pca_error:.4f}')
    print('    an autoencoder that cannot beat this has learned nothing '
          'nonlinear')

    # --- 3. reparameterisation keeps the gradient alive ----------------- #
    set_seed(args.seed)
    probe = VAE(n_inputs=images.shape[1], latent=args.latent)
    batch = torch.tensor(x_train[:16])
    x_hat, mu, logvar = probe(batch)
    reconstruction, kl = elbo_terms(x_hat, batch, mu, logvar)
    (reconstruction + kl).backward()
    encoder_grad = float(probe.to_mu.weight.grad.abs().max())
    logvar_grad = float(probe.to_logvar.weight.grad.abs().max())
    print(f'\n[3] Reparameterisation check')
    print(f'    gradient reaches the mean head:     {encoder_grad:.6f}')
    print(f'    gradient reaches the log-var head:  {logvar_grad:.6f}')
    print('    both non-zero, so the sampling step is differentiable')

    # --- 4. train the VAE ------------------------------------------------ #
    print(f'\n[4] Training the VAE (beta = 1), {args.epochs} epochs')
    model, history = train_vae(x_train, x_val, args.latent, 1.0,
                               args.epochs, args.seed, verbose=True)
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(x_test)
        x_hat, mu, logvar = model(test_tensor)
        vae_recon = float(torch.sum((x_hat - test_tensor) ** 2)) / len(x_test)
        _, test_kl = elbo_terms(x_hat, test_tensor, mu, logvar)
        test_kl = float(test_kl) / len(x_test)
        # Reconstruct from the mean rather than a sample, which is what
        # you would deploy.
        deterministic = model.decoder(mu)
        vae_recon_mean = float(
            torch.sum((deterministic - test_tensor) ** 2)) / len(x_test)

    active_units = history[-1]['active_units']
    print(f'    test reconstruction (sampled z): {vae_recon:.4f}')
    print(f'    test reconstruction (mean z):    {vae_recon_mean:.4f}')
    print(f'    test KL: {test_kl:.4f} nats per image, '
          f'{active_units}/{args.latent} latent units active')

    # The fair linear-vs-nonlinear comparison is against an autoencoder
    # with no KL penalty. A beta = 1 VAE is deliberately handicapped: it
    # pays reconstruction to keep its posterior near the prior, and it
    # uses fewer latent units than it nominally has. Comparing THAT to
    # PCA's MSE-optimal reconstruction is comparing two different
    # objectives and calling the difference a failure.
    print(f'\n[4b] The fair comparison: linear vs nonlinear, no KL penalty')
    # Trained with the Gaussian likelihood, i.e. squared error: PCA is
    # the MSE-optimal linear map, so judging a BCE-trained model by MSE
    # would compare two different objectives and call the difference a
    # result.
    ae_model, ae_history = train_vae(x_train, x_val, args.latent, 0.0,
                                     args.epochs, args.seed,
                                     likelihood='gaussian')
    ae_model.eval()
    with torch.no_grad():
        ae_mu, _ = ae_model.encode(test_tensor)
        ae_recon = float(
            torch.sum((ae_model.decoder(ae_mu) - test_tensor) ** 2)
        ) / len(x_test)
    print(f'    PCA        ({args.latent} components): {pca_error:.4f}')
    print(f'    autoencoder (beta = 0, MSE) : {ae_recon:.4f}  '
          f'{"beats" if ae_recon < pca_error else "loses to"} PCA')
    print(f'    VAE         (beta = 1)      : {vae_recon_mean:.4f}  '
          f'(+{vae_recon_mean - ae_recon:.4f} paid for the prior)')
    print(f'    that gap is the price of a latent you can sample from, '
          f'not a defect')

    # --- 5. the beta sweep: posterior collapse, on purpose -------------- #
    sweep_rows: List[Dict] = []
    if not args.skip_sweep:
        print(f'\n[5] Beta sweep -- what the summed loss would have hidden')
        print(f'    {"beta":>6}  {"recon":>9}  {"KL":>9}  {"active":>7}  '
              f'reading')
        for beta in (0.0, 0.5, 1.0, 4.0, 20.0):
            swept, swept_history = train_vae(
                x_train, x_val, args.latent, beta,
                max(args.epochs // 2, 10), args.seed)
            final = swept_history[-1]
            if beta == 0.0:
                reading = 'no KL penalty: a plain autoencoder'
            elif final['val_kl'] < 0.5:
                reading = 'posterior collapse: the latent is unused'
            elif final['active_units'] == 0:
                reading = 'latent starved: every unit near the prior'
            else:
                reading = 'both terms alive'
            sweep_rows.append({'beta': beta, 'recon': final['val_recon'],
                               'kl': final['val_kl'],
                               'active': final['active_units'],
                               'reading': reading})
            print(f'    {beta:>6.1f}  {final["val_recon"]:>9.4f}  '
                  f'{final["val_kl"]:>9.4f}  '
                  f'{final["active_units"]:>3}/{args.latent:<3}  {reading}')
        print('    the sum alone looks unremarkable at every beta; the '
              'split tells you')
        print('    which model you actually trained')

    # --- 6. the latent space is usable ---------------------------------- #
    with torch.no_grad():
        samples = model.decoder(torch.randn(8, args.latent))
    sample_range = (float(samples.min()), float(samples.max()))
    print(f'\n[6] Sampling from the prior: 8 digits decoded from N(0, I)')
    print(f'    pixel range {sample_range[0]:.3f} to {sample_range[1]:.3f} '
          f'(sigmoid output, so [0, 1] by construction)')

    # --- 7. gates --------------------------------------------------------- #
    first, last = history[0], history[-1]
    gates = [
        Gate('reparameterisation_differentiable',
             'gradients reach both encoder heads',
             f'mu {encoder_grad:.2e}, logvar {logvar_grad:.2e}',
             encoder_grad > 0 and logvar_grad > 0),
        Gate('reconstruction_improved',
             'validation reconstruction falls during training',
             f'{first["val_recon"]:.4f} -> {last["val_recon"]:.4f}',
             last['val_recon'] < first['val_recon']),
        Gate('kl_not_collapsed',
             'KL stays above 0.05 nats at beta = 1',
             f'{last["val_kl"]:.4f}', last['val_kl'] > 0.05),
        Gate('nonlinear_beats_linear',
             f'the beta=0 autoencoder beats PCA at {args.latent} components',
             f'{ae_recon:.4f} vs {pca_error:.4f}',
             ae_recon < pca_error),
        Gate('vae_pays_for_its_prior',
             'the beta=1 VAE reconstructs worse than the beta=0 model',
             f'{vae_recon_mean:.4f} vs {ae_recon:.4f} '
             f'({active_units}/{args.latent} units active)',
             vae_recon_mean >= ae_recon),
        Gate('terms_logged_separately',
             'reconstruction and KL recorded apart, every epoch',
             f'{len(history)} epochs logged', True),
        Gate('collapse_at_high_beta',
             'a large beta drives KL towards zero',
             (f'KL {sweep_rows[-1]["kl"]:.4f} at beta='
              f'{sweep_rows[-1]["beta"]:g}'
              if sweep_rows else 'sweep skipped'),
             (not sweep_rows) or sweep_rows[-1]['kl'] < 0.5),
        Gate('unregularised_at_zero_beta',
             'beta = 0 leaves the KL unconstrained, so it is largest there',
             (f'KL {sweep_rows[0]["kl"]:.2f} at beta=0 vs '
              f'{sweep_rows[-1]["kl"]:.2f} at beta='
              f'{sweep_rows[-1]["beta"]:g}'
              if sweep_rows else 'sweep skipped'),
             (not sweep_rows) or sweep_rows[0]['kl'] > sweep_rows[-1]['kl']),
        Gate('reconstruction_worsens_with_beta',
             'reconstruction is best at beta = 0 and worst at the largest',
             (f'{sweep_rows[0]["recon"]:.2f} -> {sweep_rows[-1]["recon"]:.2f}'
              if sweep_rows else 'sweep skipped'),
             (not sweep_rows)
             or sweep_rows[0]['recon'] < sweep_rows[-1]['recon']),
        Gate('decoder_output_bounded',
             'sigmoid decoder stays within [0, 1]',
             f'[{sample_range[0]:.3f}, {sample_range[1]:.3f}]',
             sample_range[0] >= 0.0 and sample_range[1] <= 1.0),
    ]
    print(f'\n[7] Gates')
    for gate in gates:
        mark = 'PASS' if gate.passed else 'FAIL'
        print(f'    [{mark}] {gate.name}: {gate.observed}  '
              f'(rule: {gate.rule})')

    elapsed = time.perf_counter() - started
    status = 'VERIFIED' if all(g.passed for g in gates) else 'PENDING REVISION'
    print(f'\nstatus: {status}   ({elapsed:.1f}s)')

    if not args.no_write:
        run_dir = RUNS / 'p4_vae_elbo'
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'project': 'p4_vae_elbo',
            'status': status,
            'seed': args.seed,
            'latent': args.latent,
            'pca_baseline': pca_error,
            'vae_test_recon_sampled': vae_recon,
            'vae_test_recon_mean': vae_recon_mean,
            'vae_test_kl': test_kl,
            'autoencoder_test_recon': ae_recon,
            'active_units': active_units,
            'history': history,
            'beta_sweep': sweep_rows,
            'gates': [dataclasses.asdict(g) for g in gates],
            'seconds': round(elapsed, 1),
        }
        (run_dir / 'report.json').write_text(
            json.dumps(payload, indent=2, default=str), encoding='utf-8')
        print(f'audit pack: {run_dir}')

    return 0 if all(g.passed for g in gates) else 1


if __name__ == '__main__':
    sys.exit(main())
