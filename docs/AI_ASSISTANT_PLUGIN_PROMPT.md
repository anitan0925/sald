# AI Assistant Prompt for Porting VA-SALD

Use this prompt when adapting VA-SALD to a new pretrained diffusion-based generator.

```text
You are helping me implement Velocity-Aware Slowly Annealed Langevin Dynamics (VA-SALD) as a plug-in sampler for my pretrained generative model.

First inspect the model and sampler code.  Identify the forward process used during pretraining, not only the inference sampler.  Write the forward Itô diffusion in the form

    dY_tau = Bbar_tau(Y_tau) d tau + sigma_bar_tau dW_tau,

and define the reverse-time family p_t = q_{T-t}, B_t = Bbar_{T-t}, sigma_t = sigma_bar_{T-t}.

Then derive the model-specific quantities needed by VA-SALD:

1. The continuity velocity u_t that transports p_t.
2. The score term grad log p_t, expressed in terms of the model output.
3. The drift B_t and diffusion scale sigma_t of the matching forward Itô process.
4. The correct conversion from the model's timestep convention to t in [0,T].
5. The Euler-Maruyama discretization for t = s/r with step size eta and K = rT/eta.

Implement VA-SALD using

    dX_s =
      [ dot_t u_t(X_s)
        + sigma_t^2/2 grad log p_t(X_s)
        - sigma_t^2/2 grad f_t(X_s) ] ds
      + sigma_t dW_s.

Equivalently, if B_t and the score are easier to access, implement

    dX_s =
      [ -dot_t B_t(X_s)
        + (dot_t + 1) sigma_t^2/2 grad log p_t(X_s)
        - sigma_t^2/2 grad f_t(X_s) ] ds
      + sigma_t dW_s.

For black-box image guidance, prefer the denoised guide schedule

    f_t(x_t) = f(x0_hat(x_t,t))

when the reward model is trained on clean images.  If the guide gradient is unavailable, implement the zeroth-order estimator

    grad f_t(x) ~= 1/(N sigma_bar_t) sum_i h(f_t(x + sigma_bar_t eps_i)) eps_i,

where eps_i are iid standard Gaussian perturbations and h is group reward normalization,

    h(R_i) = (R_i - mean_j R_j) / (std_j R_j + 1e-6).

Use the same query budget N, reward model, prompt, random seeds, image resolution, and base model when comparing against baselines.  Report any unknown forward-process parameter, such as sigma_t, as an explicit calibrated hyperparameter.

Do not insert the slowdown factor r into baselines whose algorithms do not use VA-SALD time-rescaling.  Match computational budget by matching the number of model/reward evaluations.

After implementation, add a smoke test with a very small number of steps and one prompt, then add a reproduction command for the full experiment.
```

## Common Special Cases

- VP diffusion: use the pretrained VP noise schedule and the model's noise/score prediction to recover `grad log p_t`.
- Flow matching: use the backbone velocity field and the flow-matching relation between velocity, score, and the chosen stochastic interpolant.
- Schrödinger bridge or other Itô bridges: derive `B_t`, `sigma_t`, and `u_t` from the bridge forward process before adding the guide.
