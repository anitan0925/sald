import os
import sys
import torch
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
config_flags.DEFINE_config_file("config", "config/SALD_Guidance_zerothorder.py", "Configuration.")

class Trainer:
    def __init__(self, config):
        num_gpus = torch.cuda.device_count()
        assert num_gpus == 1, "Only supports single GPU"

        self.config = config
        self.accelerator = Accelerator(
            mixed_precision="fp16",
        )
        
        self.accelerator.init_trackers(
            project_name="finetune-stable-diffusion",
            config=self.config,
            
        )
        set_seed(self.config.seed, device_specific=True)
        

        # Setup extension
        Pipeline = type('ExtendPipeline', (ExtendPipeline, StableDiffusion3Pipeline), {})
        self.pipeline = Pipeline.from_pretrained(
            self.config.diffusion.model,
            torch_dtype=torch.float32,
        )
        self.pipeline.scheduler = ExtendFlowMatchEulerDiscretScheduler.from_config(self.pipeline.scheduler.config.copy())
        self.pipeline.scheduler.init_extension(noise_level=self.config.diffusion.noise_level, std_schedule = self.config.diffusion.noise_schedule)
        
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

        

   
    def load_prompt(self):
        path = os.path.join(self.config.dataset_dir, "train.txt")
        with open(path, "r") as f:
            lines = f.readlines()
        return lines[0].strip()

    def run(self):

        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
        xk = torch.randn(1, *self.latent_shape, device=self.accelerator.device, generator=generator)
        

        ###############Test ##########
        # T = 1
        # t  in [0,1]
        # s  in [0, r] , s = r *t  , S = r *T = r 
        # K = S/ eta0 = r* T / eta0 = r / eta0 
        #########################
        r =  self.config.diffusion.r 
        eta0 = self.config.diffusion.eta0
        K = int(r/eta0)
        
        GC = self.config.diffusion.guidanceC
        
        
        t0 = 1
        Iter_Reward = np.zeros((K), dtype=np.float32) 
        for k in range(K):
            t = t0 - k*eta0/r
            sigma_t = self.config.diffusion.noise_level * t
            eta = sigma_t**2/2 *eta0
            ak = 1 -   eta/t 
            bk = eta0 /r   +   (1-t)/t *eta 
            ck =  (2*eta)**0.5   
            gk =   eta / ck  *GC 
            t_index = torch.tensor(1000*t).cuda() 
            vt = self.pipeline.flow_pred([self.prompt] * xk.shape[0], xk, t_index.expand(xk.shape[0]), self.config.diffusion.guidance_scale)
            ek_zerothorder =  torch.randn( self.config.sample.perturbation_samples, *self.latent_shape, device=self.accelerator.device, generator=generator)  #et_list[k]
            xk_perturbation =  xk   +  ck * ek_zerothorder
            print('k {:.0f},   t {:.4f},  r {:.3f}, GC {:.1f} eta {:.7f}'.format(k,t,r, GC,eta))
        
            print(self.prompt)
            print(sigma_t)
            print(self.config.seed)

            # Evaluation
            images = self.latents_to_images(xk_perturbation)
            rewards = self.reward_fn(images, [self.prompt]*xk_perturbation.shape[0]).to(self.accelerator.device)
            
            # Gradient approximation
            reward_score = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-6)
            zk = torch.einsum("N, N ... -> ...", reward_score, ek_zerothorder).unsqueeze(0) / ek_zerothorder.shape[0]  #  zeroth order gradient estimation, the term 1/ck is merged into gk

            # Update step
            ek =  torch.randn( 1, *self.latent_shape, device=self.accelerator.device, generator=generator) 
            xk = ak * xk + bk * vt + gk * zk + ck* ek


            # Test Evaluation 
            images = self.latents_to_images(xk)
            Eva_rewards = self.reward_fn(images, [self.prompt]).to(self.accelerator.device)
            
            print('Eva_Reward: {:.3f}'.format(Eva_rewards[0].data.cpu()))
            Iter_Reward[k] =Eva_rewards[0].data.cpu()
         
                    
        
        # Logging
        final_images = self.latents_to_images(xk)
        
        ImageRecords = './results/VA_SALD_' +  str(self.config.reward) +   '/VA_SALD_Guidance_zerothorder_' + self.prompt   + '_' + str(r)  + '_' + str(K) + '_' + str(eta0)  + '_' + str(GC) + '_' + str(Iter_Reward[k]) +  '_' + str(self.config.seed) +'.png'
        
        final_images[0].save(ImageRecords)

        saveRewards = './results/VA_SALD_' +  str(self.config.reward) + '/VA_SALD_Guidance_zerothorder_' + self.prompt  + '_' + str(r) + '_'   + '_' + str(self.config.seed) + '.npz'
        
        np.savez(saveRewards,array1 = Iter_Reward)  

        
        txtfile = './results/VA_SALD_' +  str(self.config.reward) + '/VA_SALD_Guidance_zerothorder_' + self.prompt   + '_' + str(r)  + '_' + str(K) + '_' + str(eta0)  + '_' + str(GC) + '.txt'
        
        with open(txtfile,"a") as f:
            f.write(str(self.config.seed)+ ' ' +  str(Iter_Reward[k]) + ' \n')

        

    def latents_to_images(self, latents):
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        images = self.pipeline.vae.decode(latents.to(self.pipeline.vae.dtype), return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type="pil")
        return images
    
    

if __name__ == "__main__":
    FLAGS(sys.argv)
    #FLAGS.config.reward = "pickscore"
    FLAGS.config.reward = "aesthetic"
    #FLAGS.config.reward = "CLIPscore"
    FLAGS.config.diffusion.r  = 1
    FLAGS.config.diffusion.guidanceC = 8
    Num_seed = 1
    for ii in range(Num_seed):
        FLAGS.config.seed = ii
        trainer = Trainer(FLAGS.config)
        trainer.run()
        trainer = []