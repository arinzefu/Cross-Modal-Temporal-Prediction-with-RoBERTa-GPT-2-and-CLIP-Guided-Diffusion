import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
from diffusers import UNet2DConditionModel, DDPMScheduler
# Overfits ONE image, then proves the model can take pure noise (or a
# partially-noised image) and reconstruct it. This is the diffusion analogue
# of your old reconstruction sanity check.
# =========================================================
# SINGLE-IMAGE DIFFUSION TEST WITH PROGRESS VISUALIZATION
# =========================================================



def single_image_diffusion_test(diffusion, dataloader, device,
                                train_steps=800, lr=2e-4, t_partial=600,
                                viz_every=100):
    model = diffusion.model

    batch = next(iter(dataloader))
    img = (batch[0] if isinstance(batch, (list, tuple)) else batch)[:1].to(device)
    img = F.interpolate(img, (224, 224), mode="bilinear", align_corners=False)
    x0 = img * 2 - 1

    to_img = lambda z: ((z[0].detach().cpu() + 1) / 2).permute(1, 2, 0).clamp(0, 1)

    # fixed noise for consistent comparison across checkpoints
    fixed_noise = torch.randn_like(x0)
    fixed_t = torch.full((1,), t_partial, device=device).long()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    print("\n===== OVERFITTING ONE IMAGE =====\n")

    for step in range(train_steps):
        t = torch.randint(0, diffusion.T, (1,), device=device).long()
        opt.zero_grad()
        loss, x0_hat = diffusion.p_losses(x0, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % viz_every == 0 or step == train_steps - 1:
            model.eval()
            with torch.no_grad():
                # noise with the SAME noise each time so panels are comparable
                x_t = diffusion.q_sample(x0, fixed_t, noise=fixed_noise)
                rec = diffusion.sample(x_start=x_t, t_start=t_partial)

                fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
                titles = ["target", f"noised @ t={t_partial}", f"denoised (step {step})"]
                for a, im, ti in zip(ax, [x0, x_t, rec], titles):
                    a.imshow(to_img(im)); a.set_title(ti); a.axis("off")
                plt.suptitle(f"step {step} | eps_mse {loss.item():.5f}", y=1.02)
                plt.tight_layout(); plt.show()
            model.train()

        elif step % 50 == 0:
            print(f"step {step:4d} | eps_mse {loss.item():.5f}")

    # ---- final: full generation from pure noise ----
    model.eval()
    with torch.no_grad():
        gen = diffusion.sample(shape=x0.shape)
        x_t = diffusion.q_sample(x0, fixed_t, noise=fixed_noise)
        rec = diffusion.sample(x_start=x_t, t_start=t_partial)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, im, ti in zip(
        ax,
        [x0, x_t, rec, gen],
        ["target", f"noised @ t={t_partial}", "denoised back", "from pure noise"],
    ):
        a.imshow(to_img(im)); a.set_title(ti); a.axis("off")
    plt.suptitle("FINAL", y=1.02); plt.tight_layout(); plt.show()
    print("===== TEST COMPLETE =====")
