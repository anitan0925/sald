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

        
        r =  self.config.diffusion.r 
        Inference_N_step = 40*r
        eta0 = self.config.diffusion.eta0
        K = int(r/eta0)
        GC = self.config.diffusion.guidanceC
        reward_name = "CLIPscore"
        # reward_name ="aesthetic"
        # reward_name ="pickscore"
        txtfile = './results/VA_SALD_' +  str(reward_name) + '/VA_SALD_Guidance_zerothorder_' + self.prompt   + '_' + str(r)  + '_' + str(K) + '_' + str(eta0)  + '_' + str(GC) + '.txt'
        # txtfile = './results/evolvable_' +  str(reward_name) + '/Evolvable_' + self.prompt  + '_' + str(Inference_N_step) + '_' + str(self.config.diffusion.guidanceC)  + '_'  + 'adjoint.txt'
        # txtfile = './results/StableDiff_FlowGRPO_DefaultSampler_' +  str(reward_name) + '/StableDiff_Zeroth_' + self.prompt  + '_' + str(Inference_N_step) + '_' + str(self.config.diffusion.guidanceC)  + '_'  + 'adjoint.txt'
        data = np.loadtxt(txtfile)


        Nscale = 30    # the scale for "CLIPscore"
        # Nscale  = 1    # the scale for "aesthetic"
        # Nscale =  26   # the scale for "pickscore"
        

        meanScore = np.mean(data[:,1]) * Nscale
        stdScore = np.std(data[:,1]) * Nscale
        
        images = []
        for k in range(10):
            ImageRecords = './results/VA_SALD_' +  str(reward_name) +   '/VA_SALD_Guidance_zerothorder_' + self.prompt   + '_' + str(r)  + '_' + str(K) + '_' + str(eta0)  + '_' + str(GC) + '_' + str(data[k,1]) +  '_' + str(k) +'.png'
            # ImageRecords = './results/evolvable_' +  str(reward_name) + '/Evolvable_' + self.prompt   +  '_' + str(Inference_N_step)  + '_' + str(self.config.diffusion.guidanceC) + '_' + str(data[k,1]) +  '_' + str(k) + '.png'
            # ImageRecords = './results/StableDiff_FlowGRPO_DefaultSampler_' +  str(reward_name) + '/StableDiff_Zeroth_' + self.prompt   +  '_' + str(Inference_N_step)  + '_' + str(self.config.diffusion.guidanceC) + '_' + str(data[k,1]) +  '_' + str(k) + '.png'
            
            img = Image.open(ImageRecords)
            images.append(img)
        

        Eva_rewards = self.reward_fn(images, [self.prompt]).to(self.accelerator.device)

        print('Mean_Reward: {:.3f}'.format(meanScore))    
        print('Std_Reward: {:.3f}'.format(stdScore))    
        print('DiversityScore: {:.3f}'.format(Eva_rewards.data.cpu()))
            
        
                    
        
        

        

    def latents_to_images(self, latents):
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        images = self.pipeline.vae.decode(latents.to(self.pipeline.vae.dtype), return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type="pil")
        return images
    
    

if __name__ == "__main__":
    FLAGS(sys.argv)
    FLAGS.config.diffusion.r = 4
    FLAGS.config.diffusion.guidanceC = 8
    FLAGS.config.reward = "DiversityScore"
    
        
    trainer = Trainer(FLAGS.config)
    trainer.run()
    
