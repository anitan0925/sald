# VA-SALD for Flow-Matching SD3.5

This directory contains the SD3.5-M implementation of Velocity-Aware SALD for the flow-matching Itô setting.  It corresponds to the guided-image-generation experiment described in the paper's additional details for Stable Diffusion 3.5 Medium.

The backbone is `stabilityai/stable-diffusion-3.5-medium`, whose pretrained dynamics are treated as a flow-matching model.  The guide is evaluated through black-box rewards such as Aesthetic Score, PickScore, and CLIP Score, and the guide gradient is estimated with zeroth-order Gaussian perturbations and group-normalized rewards.

## Method

For the flow-matching backbone, the deterministic path is written as

```math
dx_\tau = v_\tau(x_\tau)\,d\tau.
```

We use the associated stochastic process

```math
dx_\tau =
\left(
v_\tau(x_\tau) + \frac{\sigma_\tau^2}{2}\nabla\log p_\tau(x_\tau)
\right)d\tau
+ \sigma_\tau dW_\tau,
```

with the score relation

```math
\nabla\log p_\tau(x_\tau)
= -\frac{x_\tau}{\tau}
- \frac{1-\tau}{\tau}v_\tau(x_\tau).
```

With reverse time `t = 1 - tau`, slowdown `t=s/r`, and `sigma_t = (1-t) sigma_0`, VA-SALD is discretized by Euler-Maruyama.  The default configuration uses

- `sigma_0 = 0.7`
- `eta = 0.025`
- `N = 32` zeroth-order perturbations
- group reward normalization
- `r = 4` unless changed in the config

## Environment

Create the environment:

```bash
conda env create -f environment.yaml
conda activate fsd
```

The environment uses CUDA-enabled PyTorch and Diffusers.  You also need Hugging Face access to `stabilityai/stable-diffusion-3.5-medium`.

## Run

From this directory:

```bash
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=0 \
  trainer/VA_SALD_Guidance_zerothorder.py \
  --config config/VA_SALD_Guidance_zerothorder.py:base
```

The default prompt is read from:

```text
dataset/animal_one/train.txt
```

The main hyperparameters are in:

```text
config/VA_SALD_Guidance_zerothorder.py
```

Useful fields:

- `config.diffusion.r`: slowdown factor.
- `config.diffusion.eta0`: Euler-Maruyama step size in the slowed time.
- `config.diffusion.guidanceC`: guidance strength `c`.
- `config.sample.perturbation_samples`: zeroth-order query size `N`.
- `config.reward`: one of the reward names registered in `rewards.py`.

Outputs are written to `results/` by default.  These runtime outputs are ignored by git.

## Baseline Sampler

`trainer/StableDiffusion_FlowGRPOsampler_zeroth.py` implements a flow-matching zeroth-order guided sampler baseline using the same reward interface and perturbation budget.  Use it for controlled comparisons where the number of reward queries and model evaluations are matched.

## Files

```text
config/                         Experiment configs.
dataset/animal_one/             Example prompt file.
pipeline_extensions.py          SD3.5 pipeline and flow-matching scheduler extensions.
rewards.py                      Reward wrappers.
trainer/VA_SALD_Guidance_zerothorder.py
                                Flow-matching VA-SALD sampler.
trainer/StableDiffusion_FlowGRPOsampler_zeroth.py
                                Zeroth-order guided baseline.
trainer/evolvable.py            FM-Evolv-style baseline utilities.
utils.py                        Shared helpers.
```
