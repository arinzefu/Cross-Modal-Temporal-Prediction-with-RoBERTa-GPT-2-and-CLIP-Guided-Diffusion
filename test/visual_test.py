import torch
# =========================================================
# RECONSTRUCTION / SAMPLING  (verify the decoder uses the CLIP latent)
# =========================================================
@torch.no_grad()
def reconstruct(model, clip_pixels, device, image_size=64, num_inference_steps=50):
    model.eval()
    sched = DDPMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(num_inference_steps)

    clip_pixels = clip_pixels.to(device)
    cond = model.cond_proj(model.encode(clip_pixels)).unsqueeze(1)

    x = torch.randn(clip_pixels.size(0), 3, image_size, image_size, device=device)
    for t in sched.timesteps:
        noise_pred = model.unet(x, t, encoder_hidden_states=cond).sample
        x = sched.step(noise_pred, t, x).prev_sample

    x = (x.clamp(-1, 1) + 1) / 2                              # back to [0,1] for viewing
    model.train()
    return x