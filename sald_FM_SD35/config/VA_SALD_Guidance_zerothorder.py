import ml_collections

def base():
    config = ml_collections.ConfigDict()

    config.seed = 10
    config.run_name = "SALD"
    
    config.diffusion = ml_collections.ConfigDict()
    config.diffusion.model = "stabilityai/stable-diffusion-3.5-medium"
    config.diffusion.guidance_scale = 4.5
    config.diffusion.resolution = 512
    config.diffusion.noise_level = 0.7
    config.diffusion.noise_schedule =  'constant'
    

   
    config.sample = ml_collections.ConfigDict()
    config.sample.perturbation_samples = 32  # number of samples used for gradient approximation
    

    config.dataset_dir = "dataset/animal_one"
    config.reward = "aesthetic"
    
    

    

    # SALD slow down parameter r
    config.diffusion.r  = 4
    
    
    Default_base_steps = 40
    config.diffusion.eta0  = 1/Default_base_steps  ### discretization of s 
    config.diffusion.guidanceC = 8

    config.sample.diffusion_steps =  40 * config.diffusion.r

    config.dataset_dir = "dataset/animal_one"
    

    return config


def get_config(name):
    return globals()[name]()
