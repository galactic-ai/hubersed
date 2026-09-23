#!/usr/bin/env python
"""Train a spender autoencoder on DESI spectra scaled by the square root of their weights.

Redshift is set to zero for every spectrum. Run it with
``python -m hubersed.detect.train_DESI_noise DIR OUTFILE``.
"""

import argparse
import functools
import os
import time

import numpy as np
import torch
from accelerate import Accelerator
from spender import SpectrumAutoencoder
from spender.data import desi
from spender.util import mem_report
from torch import nn


def base(m):
    """Return the wrapped module if ``m`` has a ``module`` attribute, else ``m`` itself."""
    return m.module if hasattr(m, "module") else m


def prepare_train(seq, niter=800):
    """Fill in the default iteration count and encoder switches for each training mode.

    Parameters
    ----------
    seq : list of dict
        Training modes. Each dict needs a ``data`` entry and is changed in place.
    niter : int
        Iteration count given to modes that have no ``iteration`` entry.

    Returns
    -------
    list of dict
        The same list. Modes without ``encoder`` get their ``data`` entry as ``encoder``.
    """
    for d in seq:
        if "iteration" not in d:
            d["iteration"] = niter
        if "encoder" not in d:
            d.update({"encoder": d["data"]})
    return seq


def build_ladder(train_sequence):
    """Map each epoch to the index of the training mode it belongs to.

    Parameters
    ----------
    train_sequence : list of dict
        Training modes, each with an ``iteration`` count.

    Returns
    -------
    numpy.ndarray of int
        One entry per epoch holding the index of its mode in ``train_sequence``.
    """
    n_iter = sum([item["iteration"] for item in train_sequence])

    ladder = np.zeros(n_iter, dtype="int")
    n_start = 0
    for i, mode in enumerate(train_sequence):
        n_end = n_start + mode["iteration"]
        ladder[n_start:n_end] = i
        n_start = n_end
    return ladder


def get_all_parameters(models, instruments):
    """Collect optimizer parameter groups for the encoders, the shared decoder and instruments.

    The decoder parameters are taken from the first model only. Instrument parameters go in a
    second group with learning rate 1e-4, and that group is printed when it exists.

    Parameters
    ----------
    models : list of torch.nn.Module
        Autoencoders, possibly wrapped.
    instruments : list
        Instruments. Entries equal to None are skipped.

    Returns
    -------
    dicts : list of dict
        Parameter groups for a torch optimizer.
    n_parameters : int
        Number of trainable parameter elements over all groups.
    """
    model_params = []
    # multiple encoders
    for model in models:
        m = base(model)
        model_params += m.encoder.parameters()
    # 1 decoder
    model_params += base(models[0]).decoder.parameters()
    dicts = [{"params": model_params}]

    n_parameters = sum([p.numel() for p in model_params if p.requires_grad])

    instr_params = []
    # instruments
    for inst in instruments:
        if inst == None:
            continue
        instr_params += inst.parameters()
    if instr_params != []:
        dicts.append({"params": instr_params, "lr": 1e-4})
        n_parameters += sum([p.numel() for p in instr_params if p.requires_grad])
        print("parameter dict:", dicts[1])
    return dicts, n_parameters


def consistency_loss(s, s_aug, individual=False):
    """Penalise the distance between latents of original and augmented spectra.

    Parameters
    ----------
    s : torch.Tensor
        Latents of shape (batch, latent size).
    s_aug : torch.Tensor
        Latents of the augmented batch, same shape as ``s``.
    individual : bool
        Return per-spectrum values instead of the summed loss.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The summed loss, or the scaled squared distance and the per-spectrum loss when
        ``individual`` is True.
    """
    batch_size, s_size = s.shape
    x = torch.sum((s_aug - s) ** 2 / (0.5) ** 2, dim=1) / s_size
    sim_loss = torch.sigmoid(x) - 0.5  # zero = perfect alignment
    if individual:
        return x, sim_loss
    return sim_loss.sum()


def restframe_weight(model, mu=5000, sigma=2000, amp=30):
    """Return a Gaussian weight over the decoder rest-frame wavelength grid.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder, possibly wrapped.
    mu : float
        Centre of the Gaussian on the ``wave_rest`` grid.
    sigma : float
        Width parameter of the Gaussian.
    amp : float
        Peak value.

    Returns
    -------
    torch.Tensor
        Weight for each rest-frame wavelength bin.
    """
    m = base(model)
    x = m.decoder.wave_rest
    return amp * torch.exp(-((0.5 * (x - mu) / sigma) ** 2))


def similarity_restframe(
    instrument, model, s=None, slope=1.0, individual=False, wid=5, bound=[4000, 7000]
):
    """Penalise pairs whose latent distance and decoded spectrum distance disagree.

    Decoded spectra are divided by their median inside ``bound`` before the pairwise
    distances are taken. ``instrument`` is not used.

    Parameters
    ----------
    instrument : object
        Unused.
    model : torch.nn.Module
        Autoencoder, possibly wrapped.
    s : torch.Tensor
        Latents of shape (batch, latent size).
    slope : float
        Steepness of the sigmoid penalty.
    individual : bool
        Return the pairwise terms instead of the total loss.
    wid : float
        Width of the tolerated band of distance differences.
    bound : list of float
        Lower and upper rest-frame wavelength of the normalisation window.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The pairwise loss summed and divided by the batch size, or the latent distances,
        spectrum distances and pairwise loss when ``individual`` is True.
    """
    m = base(model)
    _, s_size = s.shape
    device = s.device

    spec = m.decode(s)
    wave = m.decoder.wave_rest
    mask = (wave > bound[0]) * (wave < bound[1])
    spec = spec / spec[:, mask].median(dim=1)[0][:, None]
    batch_size, spec_size = spec.shape
    # pairwise dissimilarity of spectra
    S = (spec[None, :, :] - spec[:, None, :]) ** 2
    # dissimilarity of spectra
    # of order unity, larger for spectrum pairs with more comparable bins
    W = restframe_weight(model)
    spec_sim = (W * S).sum(-1) / spec_size
    # dissimilarity of latents
    s_sim = ((s[None, :, :] - s[:, None, :]) ** 2).sum(-1) / s_size

    # only give large loss of (dis)similarities are different (either way)
    x = s_sim - spec_sim
    sim_loss = torch.sigmoid(slope * x - wid / 2) + torch.sigmoid(-slope * x - wid / 2)
    diag_mask = torch.diag(torch.ones(batch_size, device=device, dtype=bool))
    sim_loss[diag_mask] = 0

    if individual:
        return s_sim, spec_sim, sim_loss

    # The total sums N^2 terms, so divide by N to match the scale of the fidelity loss.
    return sim_loss.sum() / batch_size


def _losses(model, instrument, batch, similarity=False, slope=0, skip=False):
    """Compute the fit loss and optional similarity loss for one batch.

    The model sees ``spec * sqrt(w)`` at redshift zero. Pixels with zero weight are masked.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder, possibly wrapped.
    instrument : object
        Instrument passed to the model loss.
    batch : tuple of torch.Tensor
        Spectra, weights and redshifts.
    similarity : bool
        Add the similarity loss.
    slope : float
        Slope passed to the similarity loss.
    skip : bool
        Only encode, and return 0 for both losses.

    Returns
    -------
    loss : torch.Tensor or int
        Model loss.
    sim_loss : torch.Tensor or int
        Similarity loss, or 0.
    s : torch.Tensor
        Latents of the batch.
    """
    spec, w, z = batch

    snr = spec * torch.sqrt(w)

    # override z to z=0 since noise doesn't require redshift
    z_zero = torch.zeros_like(z)

    # need the latents later on if similarity=True
    m = base(model)
    s = m.encode(snr)

    weight = torch.ones_like(w)
    # mask out zero-weighted pixels
    weight[w == 0] = 0

    if skip:
        # used only for consistency loss; we still need s
        return 0, 0, s

    loss = m.loss(snr, weight, instrument, z=z_zero, s=s)

    # noise model doesn't use similarity
    if similarity:
        sim_loss = similarity_restframe(instrument, model, s, slope=slope)
    else:
        sim_loss = 0

    return loss, sim_loss, s


def get_losses(model, instrument, batch, aug_fct=None, similarity=True, consistency=True, slope=0):
    """Compute all loss terms for one batch, with an optional augmented copy.

    When ``aug_fct`` is given it reads ``args.z_max`` from module scope, which only exists
    when this file runs as a script.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder, possibly wrapped.
    instrument : object
        Instrument passed to the model loss.
    batch : tuple of torch.Tensor
        Spectra, weights and redshifts.
    aug_fct : callable, optional
        Function that returns an augmented copy of the batch.
    similarity : bool
        Add the similarity loss.
    consistency : bool
        Add the consistency loss when ``aug_fct`` is given.
    slope : float
        Slope for the similarity loss and scale of the consistency loss.

    Returns
    -------
    tuple
        Loss, similarity loss, augmented loss, augmented similarity loss and consistency
        loss. Terms that are switched off are 0.
    """
    loss, sim_loss, s = _losses(model, instrument, batch, similarity=similarity, slope=slope)

    if aug_fct is not None:
        batch_copy = aug_fct(batch, z_max=args.z_max)
        loss_, sim_loss_, s_ = _losses(
            model, instrument, batch_copy, similarity=similarity, slope=slope, skip=True
        )
    else:
        loss_ = sim_loss_ = 0

    if consistency and aug_fct is not None:
        cons_loss = slope * consistency_loss(s, s_)
    else:
        cons_loss = 0

    return loss, sim_loss, loss_, sim_loss_, cons_loss


def checkpoint(accelerator, args, optimizer, scheduler, n_encoder, outfile, losses):
    """Save the unwrapped model state dicts and the loss history.

    ``optimizer``, ``scheduler`` and ``n_encoder`` are accepted but not saved.

    Parameters
    ----------
    accelerator : accelerate.Accelerator
        Accelerator that prepared the models.
    args : list of torch.nn.Module
        Models to save.
    optimizer : torch.optim.Optimizer
        Unused.
    scheduler : object
        Unused.
    n_encoder : int
        Unused.
    outfile : str
        Output file path.
    losses : numpy.ndarray
        Loss history to store.
    """
    unwrapped = [accelerator.unwrap_model(args_i).state_dict() for args_i in args]

    accelerator.save(
        {
            "model": unwrapped,
            "losses": losses,
        },
        outfile,
    )
    return


def load_model(filename, models, instruments):
    """Load model weights and loss history from a checkpoint.

    Parameters
    ----------
    filename : str
        Checkpoint written by ``checkpoint``.
    models : list of torch.nn.Module
        Models that receive the weights in place.
    instruments : list
        Instruments matching ``models``. The first one sets the device.

    Returns
    -------
    models : list of torch.nn.Module
        The same models after loading.
    losses : numpy.ndarray
        Loss history stored in the checkpoint.
    """
    device = instruments[0].wave_obs.device
    model_struct = torch.load(filename, map_location=device, weights_only=False)
    # wave_rest = model_struct['model'][0]['decoder.wave_rest']
    for i, model in enumerate(models):
        # Older checkpoints name these layers mlp.mlp, so rename them to mlp.
        if "encoder.mlp.mlp.0.weight" in model_struct["model"][i].keys():
            from collections import OrderedDict

            model_struct["model"][i] = OrderedDict(
                [(k.replace("mlp.mlp", "mlp"), v) for k, v in model_struct["model"][i].items()]
            )
        # Older checkpoints lack the encoder instrument buffers, so add them and retry.
        try:
            model.load_state_dict(model_struct["model"][i], strict=False)
        except RuntimeError:
            model_struct["model"][i]["encoder.instrument.wave_obs"] = instruments[i].wave_obs
            model_struct["model"][i]["encoder.instrument.skyline_mask"] = instruments[
                i
            ].skyline_mask
            model.load_state_dict(model_struct[i]["model"], strict=False)

    losses = model_struct["losses"]
    return models, losses


def train(
    models,
    instruments,
    trainloaders,
    validloaders,
    n_epoch=200,
    outfile=None,
    losses=None,
    verbose=False,
    lr=1e-4,
    n_batch=50,
    aug_fcts=None,
    similarity=False,
    consistency=False,
):
    """Train the autoencoders and save a checkpoint every fifth epoch and at the end.

    The training modes come from ``train_sequence`` and the similarity slopes from
    ``ANNEAL_SCHEDULE``, both read from module scope, so this only runs when the file is
    executed as a script.

    Parameters
    ----------
    models : list of torch.nn.Module
        One autoencoder per instrument.
    instruments : list
        Instruments matching ``models``.
    trainloaders : list
        Training data loaders, one per instrument.
    validloaders : list
        Validation data loaders, one per instrument.
    n_epoch : int
        Number of epochs to run. Resumed training adds the epochs already stored in
        ``losses``.
    outfile : str, optional
        Checkpoint path. Defaults to ``checkpoint.pt`` in the working directory.
    losses : numpy.ndarray, optional
        Loss history from an earlier run to continue from.
    verbose : bool
        Print losses and memory use.
    lr : float
        Maximum learning rate of the one-cycle schedule.
    n_batch : int, optional
        Batches per epoch. None uses every batch.
    aug_fcts : list
        Augmentation function or None for each instrument.
    similarity : bool
        Add the similarity loss.
    consistency : bool
        Add the consistency loss.
    """
    n_encoder = len(models)
    model_parameters, n_parameters = get_all_parameters(models, instruments)

    if verbose:
        print("model parameters:", n_parameters)
        mem_report()

    ladder = build_ladder(train_sequence)
    optimizer = torch.optim.Adam(model_parameters, lr=lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)

    # Choose the accelerator, fp16 only on CUDA.
    if torch.cuda.is_available():
        accelerator = Accelerator(mixed_precision="fp16")
    else:
        accelerator = Accelerator(mixed_precision="fp16", cpu=True)  # full precision on CPU/MPS

    models = [accelerator.prepare(model) for model in models]
    instruments = [accelerator.prepare(instrument) for instrument in instruments]
    trainloaders = [accelerator.prepare(loader) for loader in trainloaders]
    validloaders = [accelerator.prepare(loader) for loader in validloaders]
    optimizer = accelerator.prepare(optimizer)

    # define losses to track
    n_loss = 5
    epoch = 0
    if losses is None:
        detailed_loss = np.zeros((2, n_encoder, n_epoch, n_loss))
    else:
        try:
            epoch = len(losses[0][0])
            n_epoch += epoch
            detailed_loss = np.zeros((2, n_encoder, n_epoch, n_loss))
            detailed_loss[:, :, :epoch, :] = losses
            if verbose:
                losses = tuple(detailed_loss[0, :, epoch - 1, :])
                vlosses = tuple(detailed_loss[1, :, epoch - 1, :])
                print(f"====> Epoch: {epoch - 1}")
                print("TRAINING Losses:", losses)
                print("VALIDATION Losses:", vlosses)
        except:  # OK if losses are empty
            pass

    if outfile is None:
        outfile = "checkpoint.pt"

    for epoch_ in range(epoch, n_epoch):
        mode = train_sequence[ladder[epoch_ - epoch]]

        # turn on/off model decoder
        for p in base(models[0]).decoder.parameters():
            p.requires_grad = mode["decoder"]

        slope = ANNEAL_SCHEDULE[(epoch_ - epoch) % len(ANNEAL_SCHEDULE)]
        if n_epoch - epoch_ <= 10:
            slope = 0  # turn off similarity

        if verbose and similarity:
            print("similarity info:", slope)

        for which in range(n_encoder):
            # turn on/off encoder
            for p in base(models[which]).encoder.parameters():
                p.requires_grad = mode["encoder"][which]

            # Skip this encoder when its dataset is switched off in this mode.
            if not mode["data"][which]:
                continue

            models[which].train()
            instruments[which].train()

            n_sample = 0
            for k, batch in enumerate(trainloaders[which]):
                batch_size = len(batch[0])
                losses = get_losses(
                    models[which],
                    instruments[which],
                    batch,
                    aug_fct=aug_fcts[which],
                    similarity=similarity,
                    consistency=consistency,
                    slope=slope,
                )
                # sum up all losses
                loss = functools.reduce(lambda a, b: a + b, losses)
                accelerator.backward(loss)
                # Clip gradients to stabilize training with the similarity loss.
                accelerator.clip_grad_norm_(model_parameters[0]["params"], 1.0)
                # once per batch
                optimizer.step()
                optimizer.zero_grad()

                # Accumulate the training losses.
                detailed_loss[0][which][epoch_] += tuple(
                    l.item() if hasattr(l, "item") else 0 for l in losses
                )
                n_sample += batch_size

                # stop after n_batch
                if n_batch is not None and k == n_batch - 1:
                    break
            detailed_loss[0][which][epoch_] /= n_sample

        scheduler.step()

        with torch.no_grad():
            for which in range(n_encoder):
                models[which].eval()
                instruments[which].eval()

                n_sample = 0
                for k, batch in enumerate(validloaders[which]):
                    batch_size = len(batch[0])
                    losses = get_losses(
                        models[which],
                        instruments[which],
                        batch,
                        aug_fct=aug_fcts[which],
                        similarity=similarity,
                        consistency=consistency,
                        slope=slope,
                    )
                    # Accumulate the validation losses.
                    detailed_loss[1][which][epoch_] += tuple(
                        l.item() if hasattr(l, "item") else 0 for l in losses
                    )
                    n_sample += batch_size

                    # stop after n_batch
                    if n_batch is not None and k == n_batch - 1:
                        break

                detailed_loss[1][which][epoch_] /= n_sample

        if verbose:
            mem_report()
            losses = tuple(detailed_loss[0, :, epoch_, :])
            vlosses = tuple(detailed_loss[1, :, epoch_, :])
            print("====> Epoch: %i" % (epoch_))
            print("TRAINING Losses:", losses)
            print("VALIDATION Losses:", vlosses)

        if epoch_ % 5 == 0 or epoch_ == n_epoch - 1:
            args = models
            checkpoint(
                accelerator,
                args,
                optimizer,
                scheduler,
                n_encoder,
                outfile,
                detailed_loss,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="data file directory")
    parser.add_argument("outfile", help="output file name")
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=2)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=512)
    parser.add_argument(
        "-l",
        "--batch_number",
        help="number of batches per epoch",
        type=int,
        default=None,
    )
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument(
        "-zmax", "--z_max", help="constrain redshifts to z_max", type=float, default=0.8
    )
    parser.add_argument("-a", "--augmentation", help="add augmentation loss", action="store_true")
    parser.add_argument("-s", "--similarity", help="add similarity loss", action="store_true")
    parser.add_argument("-c", "--consistency", help="add consistency loss", action="store_true")
    parser.add_argument(
        "-C",
        "--clobber",
        help="continue training of existing model",
        action="store_true",
    )
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    args = parser.parse_args()

    # define instruments
    instruments = [desi.DESI()]
    n_encoder = len(instruments)

    # restframe wavelength for reconstructed spectra
    # The range covers the joint dataset.
    if args.z_max > 0.01:  # DESI BGS
        lmbda_min = instruments[0].wave_obs[0] / (1.0 + args.z_max)  # 2000 A
        lmbda_max = instruments[0].wave_obs[-1]  # 9824 A
        bins = 9780
    else:  # DESI MWS
        lmbda_min = instruments[0].wave_obs[0] / (1.0 + args.z_max)
        lmbda_max = instruments[0].wave_obs[-1] / (1.0 - args.z_max)
        bins = int((lmbda_max - lmbda_min).item() / 0.8)
    wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)

    if args.verbose:
        print(f"Restframe:\t{lmbda_min:.0f} .. {lmbda_max:.0f} A ({bins} bins)")

    # data loaders
    trainloaders = [
        inst.get_data_loader(
            args.dir,
            tag="chunk1024",
            which="train",
            batch_size=args.batch_size,
            shuffle=True,
            shuffle_instance=True,
        )
        for inst in instruments
    ]
    validloaders = [
        inst.get_data_loader(
            args.dir,
            tag="chunk1024",
            which="valid",
            batch_size=args.batch_size,
            shuffle=True,
            shuffle_instance=True,
        )
        for inst in instruments
    ]

    # get augmentation function
    if args.augmentation:
        raise SystemExit("Data augmentation not implemented for noise-based training.")
    else:
        aug_fcts = [None]

    # define training sequence
    FULL = {"data": [True], "decoder": True}
    train_sequence = prepare_train([FULL])

    annealing_step = 0.1
    ANNEAL_SCHEDULE = np.arange(0.0, 2.0, annealing_step)

    if args.verbose and args.similarity:
        print("similarity_slope:", len(ANNEAL_SCHEDULE), ANNEAL_SCHEDULE)

    # define and train the model
    n_hidden = (64, 256, 1024)
    models = [
        SpectrumAutoencoder(
            instrument,
            wave_rest,
            n_latent=args.latents,
            n_hidden=n_hidden,
            act=[nn.LeakyReLU()] * (len(n_hidden) + 1),
        )
        for instrument in instruments
    ]
    # use same decoder
    if n_encoder == 2:
        models[1].decoder = models[0].decoder

    n_epoch = sum([item["iteration"] for item in train_sequence])
    init_t = time.time()
    if args.verbose:
        print("torch.cuda.device_count():", torch.cuda.device_count())
        print(f"--- Model {args.outfile} ---")

    # check if outfile already exists, continue only of -c is set
    if os.path.isfile(args.outfile) and not args.clobber:
        raise SystemExit("\nOutfile exists! Set option -C to continue training.")
    losses = None
    if os.path.isfile(args.outfile):
        if args.verbose:
            print(f"\nLoading file {args.outfile}")
        model, losses = load_model(args.outfile, models, instruments)
        non_zero = np.sum(losses[0][0], axis=1) > 0
        losses = losses[:, :, non_zero, :]

    if args.similarity:
        raise SystemExit("Noise model doesn't use similarity")

    if args.consistency:
        raise SystemExit("Noise model doesn't use consistency")

    train(
        models,
        instruments,
        trainloaders,
        validloaders,
        n_epoch=n_epoch,
        n_batch=args.batch_number,
        lr=args.rate,
        aug_fcts=aug_fcts,
        similarity=args.similarity,
        consistency=args.consistency,
        outfile=args.outfile,
        losses=losses,
        verbose=args.verbose,
    )

    if args.verbose:
        print("--- %s seconds ---" % (time.time() - init_t))
