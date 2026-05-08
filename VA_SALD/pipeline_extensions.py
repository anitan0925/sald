from typing import List, Union
import torch
from diffusers import StableDiffusion3Pipeline
import collections
from functools import wraps
import inspect
from typing import List, Union, Optional, Tuple

import pdb

def batch_cache(max_size: int = 16):
    """
    A decorator to cache responses for batch functions at an item level with an LRU policy.
    
    Arguments with a value of `None` are ignored when creating the cache key.
    
    Assumes:
    - The decorated function takes list arguments for batching.
    - All list arguments representing the batch have the same length.
    - The function returns a tuple of tensors, where the first dimension is the batch size.
    """
    cache = collections.OrderedDict()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Bind args/kwargs and find batch size
            bound_args = inspect.signature(func).bind(*args, **kwargs).arguments
            batch_size = next((len(v) for v in bound_args.values() if isinstance(v, list)), 0)
            if not batch_size: return func(*args, **kwargs)

            # 2. Create a unique key for each item, ignoring None-valued arguments
            scalar_args = tuple(sorted((k, v) for k, v in bound_args.items() if v is not None and not isinstance(v, list) ))
            list_args = {k: v for k, v in bound_args.items() if v is not None and isinstance(v, list) and len(v) == batch_size}
            keys = [scalar_args + tuple(sorted((k, v[i]) for k, v in list_args.items())) for i in range(batch_size)]

            # 3. Separate cache hits from misses
            results, miss_indices = [None] * batch_size, []
            for i, key in enumerate(keys):
                if key in cache:
                    cache.move_to_end(key)
                    results[i] = cache[key]
                else:
                    miss_indices.append(i)

            # 4. Call the original function for misses using original arguments
            if miss_indices:
                miss_kwargs = {k: [v[i] for i in miss_indices] if k in list_args else v for k, v in bound_args.items()}
                miss_results = func(**miss_kwargs)
                
                # Overrite results where 'None' is replaced by '[None]*len(miss_indices)', because our cache system assume results is always in batch
                miss_results = tuple([None]*len(miss_indices) if x is None else x for x in miss_results)

                # 5. Cache new results
                for i, original_idx in enumerate(miss_indices):
                    item_result = tuple(t[i] for t in miss_results)
                    results[original_idx] = item_result
                    cache[keys[original_idx]] = item_result
                    if len(cache) > max_size: cache.popitem(last=False)

            # 6. Reconstruct the full batch tensor output
            return tuple(torch.stack(tensors) if isinstance(tensors[0], torch.Tensor) else tensors for tensors in zip(*results) )

        return wrapper
    return decorator

class ExtendPipeline:

    def flow_pred(self, 
        prompt: List[str],
        latents: torch.FloatTensor,
        timestep: torch.FloatTensor,
        guidance_scale: float = 1.0,
        max_sequence_length: int = 256,
    ):
        do_cfg = guidance_scale != 1.0
        
        with torch.no_grad():
            (prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds) = self.encode_prompt(prompt=prompt,prompt_2=None,prompt_3=None,do_classifier_free_guidance=do_cfg,device=self._execution_device,num_images_per_prompt=1,max_sequence_length=max_sequence_length)
        
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            latents = torch.cat([latents] * 2)
            timestep = torch.cat([timestep]*2)

        
        flow_pred = - self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0].to(latents.dtype)

        if do_cfg:
            flow_pred_uncond, flow_pred_text = flow_pred.chunk(2)
            flow_pred = flow_pred_uncond + guidance_scale * (flow_pred_text - flow_pred_uncond)

        return flow_pred

import torch
import math
from contextlib import contextmanager
from typing import Optional, Tuple, Union
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerOutput
from dataclasses import dataclass

@dataclass
class ExtendFlowMatchEulerDiscretSchedulerOutput(FlowMatchEulerDiscreteSchedulerOutput):
    prev_sample_mean: torch.FloatTensor

class ExtendFlowMatchEulerDiscretScheduler(FlowMatchEulerDiscreteScheduler):

    def init_extension(self, noise_level: float = 0.7, std_schedule: str = "adjoint"):
        self.noise_level = noise_level
        self.std_schedule = std_schedule
        self.__trajectory_buffer = None


    @contextmanager
    def record_trajectory(self):
        """
        Context manager to record trajectory data during the diffusion process.
        
        Usage:
            with scheduler.record_trajectory() as trajectory:
                pipe(..., scheduler=scheduler)

        """
        data_storage = []
        self.__trajectory_buffer = data_storage
        try:
            yield data_storage
        finally:
            self.__trajectory_buffer = None
    
    
    
    def std_dev_t(self, t):
        sigma = 1-t
        if self.std_schedule == "adjoint":    # Flow-GRPO
            sigma_max = self.sigmas[1].item()
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*self.noise_level
        elif self.std_schedule == "constant": # DanceGRPO
            std_dev_t = self.noise_level
        
        return std_dev_t

    

    def get_coeficients(self, timestep):
        step_index = self.index_for_timestep(timestep)
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        t = 1-sigma
        t_next = 1-sigma_next
        dt = t_next - t
        std_dev_t = self.std_dev_t(t)

        at = 1 - (std_dev_t**2 * dt) / (2 * (1-t))
        bt = (1 + (t * (std_dev_t**2)) / (2 * (1-t))) * dt
        ct = std_dev_t * torch.sqrt(dt)
        gt =  (std_dev_t**2) / 2 * dt

        return {
            "t": t,
            "t_next": t_next,
            "std_dev_t": std_dev_t,
            "dt": dt,
            "at": at,
            "bt": bt,
            "ct": ct,
            "gt" : gt,
        }

    
class DPM2Scheduler(FlowMatchEulerDiscreteScheduler):
    
    def set_timesteps(self, *args, **kwargs):
        self.model_outputs = [None] * 2
        return super().set_timesteps(*args, **kwargs)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        
        if self.step_index is None:
            self._init_step_index(timestep)
        
        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        step_index = self.step_index
        sigma = self.sigmas[step_index]

        # 1. Convert model_output (flow/v) to data prediction x0
        # Formula from solver.py: x0_pred = sample - sigma * model_output
        x0_pred = sample - sigma * model_output

        # 2. Update history buffer
        self.model_outputs.pop(0)
        self.model_outputs.append(x0_pred)

        # 3. Solver Step
        # Use float64 for precision as in official implementation
        sigmas = self.sigmas.to(torch.float64)
        
        # Check if last step
        # The loop runs for len(sigmas) - 1 times.
        # The last step index is len(sigmas) - 2.
        lower_order_final = (step_index == len(self.sigmas) - 2)

        if step_index == 0 or lower_order_final:
            # Fallback to DDIM for the first and last step (matches solver.py logic)
            prev_sample = self._ddim_update(x0_pred, sigmas, step_index, sample)
        else:
            # DPM-Solver-2 (Multistep)
            prev_sample = self._second_order_update(self.model_outputs, sigmas, step_index, sample)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def _ddim_update(self, x0_pred, sigmas, step_index, sample):
        t, s = sigmas[step_index + 1], sigmas[step_index]
        
        # solver.py ddim_update logic with eta=0.0
        # noise_pred corresponds to (x_t - (1-s)x_0) / s
        # Note: (1-s) is alpha_s in this formulation
        noise_pred = (sample - (1 - s) * x0_pred) / s
        
        # prev_sample = (1 - t) * x0_pred + t * noise_pred
        # Note: t is next sigma, s is current sigma
        prev_sample = (1 - t) * x0_pred + t * noise_pred
        
        return prev_sample.to(sample.dtype)

    def _second_order_update(self, model_output_list, sigmas, step_index, sample):
        sigma_t = sigmas[step_index + 1]
        sigma_s0 = sigmas[step_index]
        sigma_s1 = sigmas[step_index - 1]

        alpha_t, _ = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, _ = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, _ = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h = lambda_t - lambda_s0
        h_0 = lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0 = m0
        D1 = (1.0 / r0) * (m0 - m1)

        return (
            (sigma_t / sigma_s0) * sample
            - (alpha_t * (torch.exp(-h) - 1.0)) * D0
            - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
        )