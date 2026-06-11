import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _resolve_device(model, device):
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _build_default_criterion(device):
    """Lazy import so this module has no hard dependency path on the model file."""
    try:
        from visual_autoencoder import ReconstructionLoss
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import ReconstructionLoss automatically. "
            "Pass a `criterion=` argument explicitly, e.g.\n"
            "    criterion = ReconstructionLoss(pixel_weight=0.8, perceptual_weight=0.2)\n"
            "    single_image_overfit_test(model, dl, criterion=criterion)"
        ) from e
    return ReconstructionLoss(pixel_weight=0.8, perceptual_weight=0.2).to(device)


def single_image_overfit_test(
        model,
        dataloader=None,
        *,
        image=None,
        device=None,
        criterion=None,
        n_steps=1500,
        lr=1e-4,
        log_every=50,
        viz_every=200,
        grad_clip=1.0,
        disable_noise=True,
        show_plots=True,
        verbose=True,
):
    """
    Overfit `model` to a single image and report the loss curve.

    Args:
        model:        a VisualAutoencoder (or anything with the same train/eval +
                      forward(image)->reconstruction contract).
        dataloader:   used to grab one image if `image` is not supplied. The first
                      batch element is taken: next(iter(dataloader))[0][0:1].
        image:        optional [1, 3, H, W] tensor in [0, 1] to override the dataloader.
        device:       defaults to the model's device.
        criterion:    loss module; defaults to ReconstructionLoss(0.8 pixel, 0.2 perceptual).
        n_steps:      optimisation steps.
        lr:           AdamW learning rate.
        log_every:    print interval (0 to disable).
        viz_every:    original-vs-reconstruction plot interval (0 to disable).
        grad_clip:    max grad norm (None to disable).
        disable_noise: temporarily set z/spatial noise std to 0 for a true capacity
                      test, then restore. Set False to keep the robustness noise on.
        show_plots:   render matplotlib figures (set False for headless runs).
        verbose:      print step logs.

    Returns:
        loss_history (list[float])
    """
    device = _resolve_device(model, device)

    # ----- get ONE image -----
    if image is None:
        if dataloader is None:
            raise ValueError("Provide either `image` or `dataloader`.")
        image = next(iter(dataloader))[0][0:1]
    single_img = image.to(device)

    # Resize to CLIP resolution (the encoder also does this internally; kept so the
    # displayed "Original" matches exactly what is fed in).
    single_img = F.interpolate(
        single_img,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    # ----- temporarily disable robustness noise for a clean capacity test -----
    saved_noise = {}
    if disable_noise:
        for attr in ("z_noise_std", "spatial_noise_std"):
            if hasattr(model, attr):
                saved_noise[attr] = getattr(model, attr)
                setattr(model, attr, 0.0)

    was_training = model.training

    # ----- optimizer / loss -----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer_test = torch.optim.AdamW(trainable, lr=lr)

    if criterion is None:
        criterion = _build_default_criterion(device)

    loss_history = []

    try:
        model.train()

        for step in range(n_steps):
            optimizer_test.zero_grad()

            reconstructed = model(single_img)
            loss = criterion(reconstructed, single_img)

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

            optimizer_test.step()
            loss_history.append(loss.item())

            # ----- logging -----
            if verbose and log_every and step % log_every == 0:
                print(f"Step {step}/{n_steps} | Loss: {loss.item():.4f}")

            # ----- visualization -----
            if show_plots and viz_every and step % viz_every == 0:
                model.eval()
                with torch.no_grad():
                    recon = model(single_img)

                fig, ax = plt.subplots(1, 2, figsize=(10, 5))

                ax[0].imshow(
                    single_img[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
                )
                ax[0].set_title("Original")
                ax[0].axis("off")

                ax[1].imshow(
                    recon[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
                )
                ax[1].set_title(f"Step {step}")
                ax[1].axis("off")

                plt.show()

                model.train()

        # ----- final loss curve -----
        if show_plots:
            plt.figure(figsize=(8, 5))
            plt.plot(loss_history)
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title("Single Image Overfit Test")
            plt.show()

    finally:
        # Restore noise settings and the model's original train/eval mode,
        # even if the loop is interrupted.
        for attr, val in saved_noise.items():
            setattr(model, attr, val)
        model.train(was_training)

    return loss_history