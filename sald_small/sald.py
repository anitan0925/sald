import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import safetensors.torch
from diffusers import DDPMPipeline, DDPMScheduler


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_unconditional_ddpm(model_id):
    pipe = DDPMPipeline.from_pretrained(model_id).to(device)
    print("loaded:", model_id)
    print("device:", device)
    print("sample_size:", pipe.unet.config.sample_size)
    print("in_channels:", pipe.unet.config.in_channels)
    print("prediction_type:", pipe.scheduler.config.prediction_type)
    print("num_train_timesteps:", pipe.scheduler.config.num_train_timesteps)
    return pipe

problem="mnist"
#problem="bedroom"
#problem="celeb"

if problem=="mnist":
    pipe = load_unconditional_ddpm("1aurent/ddpm-mnist")
elif problem=="bedroom":
    pipe = load_unconditional_ddpm("google/ddpm-ema-bedroom-256")
elif problem=="celeb":
    pipe = load_unconditional_ddpm("google/ddpm-ema-celebahq-256") 
else:
    print("Choose the problem: mnist, bedroom")
    exit(1)


def tensor_to_pil(x):
    """
    x: [B,C,H,W] in [-1,1]
    """
    x = (x / 2 + 0.5).clamp(0, 1)
    x = x.detach().cpu()
    images = []
    for img in x:
        img = img.permute(1, 2, 0).float().numpy()
        if img.shape[-1] == 1:
            img = (img[..., 0] * 255).astype(np.uint8)
            images.append(Image.fromarray(img, mode="L"))
        else:
            img = (img * 255).astype(np.uint8)
            images.append(Image.fromarray(img))
    return images

def show_images(images, titles=None, figsize=(12, 4)):
    n = len(images)
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(1, n, i + 1)
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        cmap = "gray" if arr.ndim == 2 else None
        plt.imshow(arr, cmap=cmap)
        plt.axis("off")
        if titles is not None:
            plt.title(titles[i])
    plt.show()

def sample_initial_noise(pipe, batch_size=4, seed=123):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (
            batch_size,
            pipe.unet.config.in_channels,
            pipe.unet.config.sample_size,
            pipe.unet.config.sample_size,
        ),
        generator=generator,
        device=device,
        dtype=next(pipe.unet.parameters()).dtype,
    )
    return x, generator

#generator = torch.Generator(device=device).manual_seed(123)
#out = pipe(batch_size=4, num_inference_steps=100, generator=generator)
#show_images(out.images, titles=[f"baseline {i}" for i in range(4)])
#for i, img in enumerate(out.images):
#    img.save(f"./baseline_{i}.png")

def slowdown_linear_factory(r: float):
    if r < 1:
        raise ValueError("r must be > 1")

    def slowdown_fn(s: torch.Tensor):
        return s / r

    return slowdown_fn


def interpolate_model_timestep(timesteps: torch.Tensor, u: torch.Tensor):
    """
    timesteps: descending tensor of shape [N]
    u: scalar in [0,1]
    """
    n = timesteps.shape[0]
    pos = torch.clamp(u, 0.0, 1.0) * (n - 1)

    i0 = torch.floor(pos).long()
    i1 = torch.clamp(i0 + 1, max=n - 1)
    w = (pos - i0.float()).to(timesteps.dtype)

    t0 = timesteps[i0]
    t1 = timesteps[i1]
    return (1.0 - w) * t0 + w * t1

def interpolate_alpha_bar(alphas_cumprod: torch.Tensor, t_float: torch.Tensor):
    T = alphas_cumprod.shape[0]
    t_float = torch.clamp(t_float, 0.0, T - 1.0)

    i0 = torch.floor(t_float).long()
    i1 = torch.clamp(i0 + 1, max=T - 1)
    w = (t_float - i0.float()).to(alphas_cumprod.dtype)

    a0 = alphas_cumprod[i0]
    a1 = alphas_cumprod[i1]
    return (1.0 - w) * a0 + w * a1

def sigma_from_t_float(alphas_cumprod: torch.Tensor, t_float: torch.Tensor, eps: float = 1e-5):
    alpha_bar = interpolate_alpha_bar(alphas_cumprod, t_float)
    return torch.sqrt(torch.clamp(1.0 - alpha_bar, min=eps))


def make_eta_schedule(
    schedule_type="constant",
    eta0=0.005,
    alpha=1.0,
    sigma_min=0.05,
):
    """
    Returns a function eta_k = eta_schedule(s_k, u_k, sigma_k).
    """

    if eta0 <= 0:
        raise ValueError("eta0 must be > 0")

    if schedule_type == "constant":
        def eta_schedule(s_k, u_k, sigma_k):
            return torch.as_tensor(eta0, device=s_k.device, dtype=s_k.dtype)

    elif schedule_type == "time_decay":
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        def eta_schedule(s_k, u_k, sigma_k):
            return torch.as_tensor(
                eta0 / (1.0 + alpha * float(s_k)),
                device=s_k.device,
                dtype=s_k.dtype,
            )

    elif schedule_type == "sigma_aware":
        if sigma_min <= 0:
            raise ValueError("sigma_min must be > 0")
        def eta_schedule(s_k, u_k, sigma_k):
            sigma_eff = torch.clamp(sigma_k, min=sigma_min)
            return torch.as_tensor(
                eta0 * float(sigma_eff ** 2),
                device=s_k.device,
                dtype=s_k.dtype,
            )

    elif schedule_type == "time_decay_sigma":
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if sigma_min <= 0:
            raise ValueError("sigma_min must be > 0")
        def eta_schedule(s_k, u_k, sigma_k):
            sigma_eff = torch.clamp(sigma_k, min=sigma_min)
            val = (eta0 / (1.0 + alpha * float(s_k))) * float(sigma_eff ** 2)
            return torch.as_tensor(val, device=s_k.device, dtype=s_k.dtype)

    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    return eta_schedule

def make_gamma_schedule(
    schedule_type="poly_decay",
    gamma_max=1.0,
    power=1.0,
):
    """
    gamma is a function of normalized time u in [0,1].
    Must satisfy gamma(1)=0 for your intended design.
    """

    if schedule_type == "zero":
        def gamma_fn(u):
            return torch.zeros_like(u)

    elif schedule_type == "constant":
        def gamma_fn(u):
            return gamma_max * torch.ones_like(u)

    elif schedule_type == "poly_decay":
        if power <= 0:
            raise ValueError("power must be > 0")
        def gamma_fn(u):
            return gamma_max * torch.clamp(1.0 - u, min=0.0) ** power

    elif schedule_type == "cosine_decay":
        def gamma_fn(u):
            z = torch.clamp(u, 0.0, 1.0)
            return gamma_max * 0.5 * (1.0 + torch.cos(math.pi * z))

    else:
        raise ValueError(f"Unknown gamma schedule: {schedule_type}")

    return gamma_fn

def make_mobility_schedule(
    schedule_type="constant",
    a0=1.0,
    a_min=0.0,
    power=1.0,
):
    """
    Returns mobility a(u), where u in [0,1].
    """

    if a0 < 0:
        raise ValueError("a0 must be >= 0")
    if a_min < 0:
        raise ValueError("a_min must be >= 0")

    if schedule_type == "constant":
        def mobility_fn(u):
            return a0 * torch.ones_like(u)

    elif schedule_type == "poly_decay":
        if power <= 0:
            raise ValueError("power must be > 0")
        def mobility_fn(u):
            z = torch.clamp(1.0 - u, min=0.0)
            return a_min + (a0 - a_min) * (z ** power)

    elif schedule_type == "cosine_decay":
        def mobility_fn(u):
            z = torch.clamp(u, 0.0, 1.0)
            return a_min + (a0 - a_min) * 0.5 * (1.0 + torch.cos(math.pi * z))

    else:
        raise ValueError(f"Unknown mobility schedule: {schedule_type}")

    return mobility_fn


def make_beta_mobility_schedule_from_scheduler(pipe, multiply_by_num_steps=True):
    betas = pipe.scheduler.betas.to(device=device)
    N = betas.shape[0]

    def mobility_fn(u):
        # u=0: noisy side, u=1: data side
        pos = torch.clamp(u, 0.0, 1.0) * (N - 1)

        # reverse orientation
        pos_rev = (N - 1) - pos

        i0 = torch.floor(pos_rev).long()
        i1 = torch.clamp(i0 + 1, max=N - 1)
        w = (pos_rev - i0.float()).to(betas.dtype)

        b0 = betas[i0]
        b1 = betas[i1]
        beta_rev = (1.0 - w) * b0 + w * b1

        
        if multiply_by_num_steps:
            return 0.5*(N - 1) * beta_rev
        else:
            return 0.5*beta_rev

    return mobility_fn


@torch.no_grad()
def sald(
    pipe,
    batch_size=1,
    num_inference_steps=100,   # pretrained model time grid
    seed=123,
    slowdown_fn=None,          # u = t(s)
    s_end=1.0,
    predictor=False,
    eta_schedule=None,         # eta_k = eta_schedule(s_k, u_k, sigma_k)
    gamma_schedule=None,       # gamma(u)
    mobility_schedule=None,
    u_max=0.98,
    max_steps=20000,
    return_all=False,
    verbose=False,
):
    unet = pipe.unet
    scheduler = pipe.scheduler

    if scheduler.config.prediction_type != "epsilon":
        raise ValueError(
            f"sald assumes prediction_type='epsilon', "
            f"got {scheduler.config.prediction_type}"
        )

    if slowdown_fn is None:
        slowdown_fn = identity_time_scale()

    if eta_schedule is None:
        eta_schedule = make_eta_schedule(schedule_type="constant", eta0=0.005)

    if gamma_schedule is None:
        gamma_schedule = make_gamma_schedule(schedule_type="zero")

    if mobility_schedule is None:
        mobility_schedule = make_mobility_schedule(schedule_type="constant", a0=1.0)
      

    x, generator = sample_initial_noise(pipe, batch_size=batch_size, seed=seed)

    scheduler.set_timesteps(num_inference_steps, device=device)
    base_timesteps = scheduler.timesteps.to(device=device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=x.dtype)

    traj = [x.detach().cpu()] if return_all else None

    s_k = torch.tensor(0.0, device=device, dtype=x.dtype)
    step = 0

    if verbose:
        print(f"Start moving-target SALD: s_end={s_end}, u_max={u_max}, max_steps={max_steps}")

    while step < max_steps:
        u_k = torch.clamp(slowdown_fn(s_k), 0.0, u_max)

        t_k = interpolate_model_timestep(base_timesteps.float(), u_k).to(device=device, dtype=x.dtype)
        sigma_k = sigma_from_t_float(alphas_cumprod, t_k).to(device=device, dtype=x.dtype)
        eta_k = eta_schedule(s_k, u_k, sigma_k).to(device=device, dtype=x.dtype)
   
        if float(eta_k) <= 0:
            raise ValueError("eta_schedule returned non-positive eta_k")

        gamma_k = gamma_schedule(u_k).to(device=device, dtype=x.dtype)
        a_k = mobility_schedule(u_k).to(device=device, dtype=x.dtype)
        if float(a_k) < 0:
            raise ValueError("mobility_schedule returned negative mobility")
        
        model_input = scheduler.scale_model_input(x, t_k)
        eps = unet(model_input, t_k).sample

        # score of p_t
        base_score = -eps / sigma_k

        # score of pi_t without guide
        if predictor:
            score_pi = (1.0 + 1.0/s_end) * base_score + (gamma_k / s_end) * x
        else:
            score_pi = base_score
            
        z = torch.randn(x.shape, generator=generator, device=x.device,dtype=x.dtype)
        
        drift = eta_k * a_k * score_pi
        diffusion = torch.sqrt(2.0 * eta_k * a_k) * z
        x = x + drift + diffusion

        if verbose and (step % 20 == 0):
            drift_norm = drift.flatten(1).norm(dim=1).mean().item()
            diff_norm = diffusion.flatten(1).norm(dim=1).mean().item()
            print(
                f"[step {step:05d}] "
                f"s={float(s_k):.4f}, u={float(u_k):.4f}, "
                f"t={float(t_k):.2f}, sigma={float(sigma_k):.4f}, "
                f"eta={float(eta_k):.6f}, gamma={float(gamma_k):.6f}, "
                f"a={a_k}, "
                f"drift_norm={drift_norm:.4f}, noise_norm={diff_norm:.4f}"
            )

        if return_all:
            traj.append(x.detach().cpu())

        s_k = s_k + eta_k
        step += 1

        if float(s_k) >= s_end:
            break
        if float(u_k) >= u_max:
            break

    if verbose:
        print(f"Finished at step={step}, final s={float(s_k):.4f}, final u={float(u_k):.4f}")

    images = tensor_to_pil(x)
    if return_all:
        return images, traj
    return images

if problem=="mnist":
    r = 10.0
    slowdown_fn = slowdown_linear_factory(r)

    eta_schedule = make_eta_schedule(
        #schedule_type="time_decay",
        schedule_type="constant",
        eta0=0.01,
        alpha=1.0)

    gamma_schedule = make_gamma_schedule(
        #schedule_type="zero",
        #schedule_type="poly_decay",
        schedule_type="constant",
        gamma_max=1.0,
        power=1.0)
    
    imgs_mnist = sald(
        pipe,
        batch_size=4,
        num_inference_steps=1000,
        seed=123,
        slowdown_fn=slowdown_fn,
        s_end=r,
        predictor=False,
        eta_schedule=eta_schedule,
        gamma_schedule=gamma_schedule,
        u_max=1.0,
        max_steps=5000,
        verbose=True)

    for i, img in enumerate(imgs_mnist):
        img.save(f"./mnist_r{r}_{i}.png")
        
elif problem in ["bedroom", "celeb"]:
    r = 10.0
    slowdown_fn = slowdown_linear_factory(r)

    
    eta_schedule = make_eta_schedule(
        schedule_type="constant",
        #schedule_type="time_decay",
        eta0 = 0.005 )

    mobility_schedule = make_mobility_schedule(schedule_type="poly_decay", a0=10.0, a_min=0.1, power=1.0)
    #mobility_schedule = make_beta_mobility_schedule_from_scheduler(pipe)
    #mobility_schedule = make_mobility_schedule(schedule_type="constant", a0=1.0)

    gamma_schedule = make_gamma_schedule(
        #schedule_type="poly_decay",
        schedule_type="constant",
        gamma_max=1.0,
        power=0.2)

    imgs = sald(
        pipe,
        batch_size=1,
        num_inference_steps=1000,
        seed=123,
        slowdown_fn=slowdown_fn,
        s_end=r,
        predictor=True,
        eta_schedule=eta_schedule,
        gamma_schedule=gamma_schedule,
        mobility_schedule=mobility_schedule,
        u_max=1.0,
        max_steps=100000,
        verbose=True)

    
    imgs[0].save(f"./{problem}_r{r}.png")
