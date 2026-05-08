import os
import sys
import torch
#import wandb
import numpy as np
from PIL import Image
from tqdm import tqdm
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusion3Pipeline
from pipeline_extensions import ExtendPipeline, ExtendFlowMatchEulerDiscretScheduler
from rewards import REWARDS_CLS
import einops
import tempfile
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from pathlib import Path

import pdb

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/StableDiff_Zeroth.py", "Configuration.")

class Trainer:
    def __init__(self, config):
        num_gpus = torch.cuda.device_count()
        assert num_gpus == 1, "Only supports single GPU"

        self.config = config
        self.accelerator = Accelerator(
            # log_with="wandb",
            mixed_precision="fp16",
        )
        
        self.accelerator.init_trackers(
            project_name="finetune-stable-diffusion",
            config=self.config,
            # init_kwargs={"wandb": {"name": self.config.run_name, "config": self.config.to_dict()}}
        )
        set_seed(self.config.seed, device_specific=True)

        # Setup extension
        Pipeline = type('ExtendPipeline', (ExtendPipeline, StableDiffusion3Pipeline), {})
        self.pipeline = Pipeline.from_pretrained(
            self.config.diffusion.model,
            torch_dtype=torch.float32,
        )
        self.pipeline.scheduler = ExtendFlowMatchEulerDiscretScheduler.from_config(self.pipeline.scheduler.config.copy())
        self.pipeline.scheduler.init_extension(noise_level=self.config.diffusion.noise_level)
        
        # Optimize model
        self.pipeline.transformer.to(torch.float16)
        self.pipeline.to(self.accelerator.device)
        self.pipeline.vae.enable_slicing()
        self.accelerator.prepare(self.pipeline.transformer)
        del self.pipeline.vae.encoder
        
        # Disable all gradients
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
        self.pipeline.text_encoder_3.requires_grad_(False)
        self.pipeline.transformer.requires_grad_(False)

        # Setup reward
        RewardCls = REWARDS_CLS[self.config.reward]
        self.reward_fn = RewardCls()
        self.reward_fn.to(self.accelerator.device)

        # Setup data
        self.prompt = self.load_prompt()

        # Utils
        self.latent_shape = (self.pipeline.transformer.config.in_channels, self.config.diffusion.resolution // self.pipeline.vae_scale_factor, self.config.diffusion.resolution // self.pipeline.vae_scale_factor)

        #self.log_code()

    def log_code(self):

        if not self.accelerator.is_main_process:
            return

        cwd = os.path.abspath(os.getcwd())
        imported_py_files = set()
        for module in sys.modules.values():
            path = getattr(module, "__file__", None)
            if path and path.endswith(".py"):
                abs_path = os.path.abspath(path)
                if abs_path.startswith(cwd):
                    imported_py_files.add(abs_path)

        self.accelerator.get_tracker("wandb").run.log_code(".", include_fn=lambda path: path in imported_py_files)

    def load_prompt(self):
        path = os.path.join(self.config.dataset_dir, "train.txt")
        with open(path, "r") as f:
            lines = f.readlines()
        return lines[0].strip()

    def run(self):

        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
        xt = torch.randn(1, *self.latent_shape, device=self.accelerator.device, generator=generator)
        

        

        
        timesteps, _ = retrieve_timesteps(self.pipeline.scheduler, num_inference_steps=self.config.sample.diffusion_steps, device=self.accelerator.device)
        Inference_N_step = timesteps.shape[0]
        Iter_Reward = np.zeros((Inference_N_step), dtype=np.float32)  
        
        for i, timestep in enumerate(timesteps):
            coe = self.pipeline.scheduler.get_coeficients(timestep)
            
            vt = self.pipeline.flow_pred([self.prompt] * xt.shape[0], xt, timestep.expand(xt.shape[0]), self.config.diffusion.guidance_scale)
            et_zerothorder =  torch.randn( self.config.sample.perturbation_samples, *self.latent_shape, device=self.accelerator.device, generator=generator)  #et_list[k]
            xt_perturbation =  xt   +  coe["ct"]  * et_zerothorder


            # Evaluation
            images = self.latents_to_images(xt_perturbation)
            rewards = self.reward_fn(images, [self.prompt]*xt_perturbation.shape[0]).to(self.accelerator.device)
            
            # Gradient approximation
            reward_score = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-6)
            zt = torch.einsum("N, N ... -> ...", reward_score, et_zerothorder).unsqueeze(0) / et_zerothorder.shape[0]  #  zeroth order gradient estimation, the term 1/ck is merged into gk

            # Diffusion step
            et =  torch.randn( 1, *self.latent_shape, device=self.accelerator.device, generator=generator) #ek_list[k]
            gt = self.config.diffusion.guidanceC* coe["gt"] /coe["ct"] 
            xt =  coe["at"] * xt + coe["bt"] * vt + gt * zt + coe["ct"]* et


            # Test Evaluation 
            images = self.latents_to_images(xt)
            Eva_rewards = self.reward_fn(images, [self.prompt]).to(self.accelerator.device)
            
            print('i: {:.0f}  Eva_Reward: {:.3f}  NumStep: {:.0f} GC: {:.1f}'.format(i, Eva_rewards[0].data.cpu(), Inference_N_step,self.config.diffusion.guidanceC))
            print(self.prompt)
            print(self.config.seed)
            
            Iter_Reward[i] =Eva_rewards[0].data.cpu()

            
            
        # Logging
        final_images = self.latents_to_images(xt)
        
        ImageRecords = './final_results11/StableDiff_FlowGRPO_DefaultSampler_' +  str(self.config.reward) + '/StableDiff_Zeroth_' + self.prompt   +  '_' + str(Inference_N_step)  + '_' + str(self.config.diffusion.guidanceC) + '_' + str(Eva_rewards[0].data.cpu().numpy()) +  '_' + str(self.config.seed) + '.png'
        
        
        final_images[0].save(ImageRecords)

        saveRewards = './final_results11/StableDiff_FlowGRPO_DefaultSampler_' +  str(self.config.reward) + '/StableDiff_Zeroth_' + self.prompt  + '_' + str(Inference_N_step) + '_' + str(self.config.diffusion.guidanceC)  +   '_' + str(self.config.seed) + '.npz'
        
        np.savez(saveRewards,array1 = Iter_Reward)  

        txtfile = './final_results11/StableDiff_FlowGRPO_DefaultSampler_' +  str(self.config.reward) + '/StableDiff_Zeroth_' + self.prompt  + '_' + str(Inference_N_step) + '_' + str(self.config.diffusion.guidanceC)  + '_' + str(self.config.diffusion.noise_schedule) + '.txt'
        
        with open(txtfile,"a") as f:
            f.write(str(self.config.seed)+ ' ' +  str(str(Eva_rewards[0].data.cpu().numpy())) + ' \n')    

        
        

    def latents_to_images(self, latents):
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        images = self.pipeline.vae.decode(latents.to(self.pipeline.vae.dtype), return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type="pil")
        return images
    
    

if __name__ == "__main__":
    FLAGS(sys.argv)
    scores = ["CLIPscore","aesthetic", "pickscore" ]
    FLAGS.config.diffusion.r = 4
    FLAGS.config.diffusion.guidanceC = 8 
    Num_seed = 10
    for jj in range(3):
        FLAGS.config.reward = scores[jj]
        for ii in range(Num_seed):
            FLAGS.config.seed = ii
            trainer = Trainer(FLAGS.config)
            trainer.run()
            trainer = []