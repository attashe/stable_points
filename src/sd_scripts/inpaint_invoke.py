
import os
import math
import random
import torch
import numpy as  np
from tqdm import tqdm, trange
from PIL import Image
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
# from ldm.util import rand_perlin_2d
# from ldm.invoke.devices             import choose_autocast
# from ldm.invoke.generator.img2img   import Img2Img
from ldm.models.diffusion.ddim     import DDIMSampler
# from ldm.models.diffusion.ksampler import KSampler
# from ldm.invoke.generator.base      import Generator


def rand_perlin_2d(shape, res, device, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1]), indexing='ij'), dim = -1).to(device) % 1

    rand_val = torch.rand(res[0]+1, res[1]+1)
    
    angles = 2*math.pi*rand_val
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1).to(device)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)

    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0]).to(device)
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0]).to(device)
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1]).to(device)
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1]).to(device)
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]).to(device)


downsampling = 8

class Generator():
    def __init__(self, model, precision):
        self.model               = model
        self.precision           = precision
        self.seed                = None
        self.latent_channels     = model.channels
        self.downsampling_factor = downsampling   # BUG: should come from model or config
        self.perlin              = 0.0
        self.threshold           = 0
        self.variation_amount    = 0
        self.with_variations     = []

    # this is going to be overridden in img2img.py, txt2img.py and inpaint.py
    def get_make_image(self,prompt,**kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        """
        raise NotImplementedError("image_iterator() must be implemented in a descendent class")

    def set_variation(self, seed, variation_amount, with_variations):
        self.seed             = seed
        self.variation_amount = variation_amount
        self.with_variations  = with_variations

    def generate(self,prompt,init_image,width,height,iterations=1,seed=None,
                 image_callback=None, step_callback=None, threshold=0.0, perlin=0.0,
                 **kwargs):
        scope = torch.autocast("cuda")
        make_image          = self.get_make_image(
            prompt,
            init_image    = init_image,
            width         = width,
            height        = height,
            step_callback = step_callback,
            threshold     = threshold,
            perlin        = perlin,
            **kwargs
        )

        results             = []
        seed                = seed if seed is not None else self.new_seed()
        first_seed          = seed
        seed, initial_noise = self.generate_initial_noise(seed, width, height)
        # with scope(self.model.device.type), self.model.ema_scope():
        with torch.autocast("cuda"):
            for n in trange(iterations, desc='Generating'):
                x_T = None
                if self.variation_amount > 0:
                    seed_everything(seed)
                    target_noise = self.get_noise(width,height)
                    x_T = self.slerp(self.variation_amount, initial_noise, target_noise)
                elif initial_noise is not None:
                    # i.e. we specified particular variations
                    x_T = initial_noise
                else:
                    seed_everything(seed)
                    try:
                        x_T = self.get_noise(width,height)
                    except:
                        pass

                image = make_image(x_T)
                results.append([image, seed])
                if image_callback is not None:
                    image_callback(image, seed, first_seed=first_seed)
                seed = self.new_seed()
        return results
    
    def sample_to_image(self,samples):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        """
        x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        if len(x_samples) != 1:
            raise Exception(
                f'>> expected to get a single image, but got {len(x_samples)}')
        x_sample = 255.0 * rearrange(
            x_samples[0].cpu().numpy(), 'c h w -> h w c'
        )
        return Image.fromarray(x_sample.astype(np.uint8))

    def generate_initial_noise(self, seed, width, height):
        initial_noise = None
        if self.variation_amount > 0 or len(self.with_variations) > 0:
            # use fixed initial noise plus random noise per iteration
            seed_everything(seed)
            initial_noise = self.get_noise(width,height)
            for v_seed, v_weight in self.with_variations:
                seed = v_seed
                seed_everything(seed)
                next_noise = self.get_noise(width,height)
                initial_noise = self.slerp(v_weight, initial_noise, next_noise)
            if self.variation_amount > 0:
                random.seed() # reset RNG to an actually random state, so we can get a random seed for variations
                seed = random.randrange(0,np.iinfo(np.uint32).max)
            return (seed, initial_noise)
        else:
            return (seed, None)

    # returns a tensor filled with random numbers from a normal distribution
    def get_noise(self,width,height):
        """
        Returns a tensor filled with random numbers, either form a normal distribution
        (txt2img) or from the latent image (img2img, inpaint)
        """
        raise NotImplementedError("get_noise() must be implemented in a descendent class")
    
    def get_perlin_noise(self,width,height):
        fixdevice = 'cpu' if (self.model.device.type == 'mps') else self.model.device
        return torch.stack([rand_perlin_2d((height, width), (8, 8), device = self.model.device).to(fixdevice) for _ in range(self.latent_channels)], dim=0).to(self.model.device)
    
    def new_seed(self):
        self.seed = random.randrange(0, np.iinfo(np.uint32).max)
        return self.seed

    def slerp(self, t, v0, v1, DOT_THRESHOLD=0.9995):
        '''
        Spherical linear interpolation
        Args:
            t (float/np.ndarray): Float value between 0.0 and 1.0
            v0 (np.ndarray): Starting vector
            v1 (np.ndarray): Final vector
            DOT_THRESHOLD (float): Threshold for considering the two vectors as
                                colineal. Not recommended to alter this.
        Returns:
            v2 (np.ndarray): Interpolation vector between v0 and v1
        '''
        inputs_are_torch = False
        if not isinstance(v0, np.ndarray):
            inputs_are_torch = True
            v0 = v0.detach().cpu().numpy()
        if not isinstance(v1, np.ndarray):
            inputs_are_torch = True
            v1 = v1.detach().cpu().numpy()

        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(self.model.device)

        return v2

    # this is a handy routine for debugging use. Given a generated sample,
    # convert it into a PNG image and store it at the indicated path
    def save_sample(self, sample, filepath):
        image = self.sample_to_image(sample)
        dirname = os.path.dirname(filepath) or '.'
        if not os.path.exists(dirname):
            print(f'** creating directory {dirname}')
            os.makedirs(dirname, exist_ok=True)
        image.save(filepath,'PNG')


class Img2Img(Generator):
    def __init__(self, model, precision):
        super().__init__(model, precision)
        self.init_latent         = None    # by get_noise()

    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,init_image,strength,step_callback=None,threshold=0.0,perlin=0.0,**kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it.
        """
        self.perlin = perlin

        sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
        )

        # scope = choose_autocast(self.precision)
        with torch.autocast("cuda"):
            self.init_latent = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(init_image)
            ) # move to latent space

        t_enc = int(strength * steps)
        uc, c   = conditioning

        def make_image(x_T):
            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                self.init_latent,
                torch.tensor([t_enc]).to(self.model.device),
                noise=x_T
            )
            # decode it
            samples = sampler.decode(
                z_enc,
                c,
                t_enc,
                img_callback = step_callback,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
                init_latent = self.init_latent,  # changes how noising is performed in ksampler
            )

            return self.sample_to_image(samples)

        return make_image

    def get_noise(self,width,height):
        device      = self.model.device
        init_latent = self.init_latent
        assert init_latent is not None,'call to get_noise() when init_latent not set'
        if device.type == 'mps':
            x = torch.randn_like(init_latent, device='cpu').to(device)
        else:
            x = torch.randn_like(init_latent, device=device)
        if self.perlin > 0.0:
            shape = init_latent.shape
            x = (1-self.perlin)*x + self.perlin*self.get_perlin_noise(shape[3], shape[2])
        return x


class Inpaint(Img2Img):
    def __init__(self, model, precision):
        self.init_latent = None
        super().__init__(model, precision)

    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,init_image,mask_image,strength,
                       step_callback=None,inpaint_replace=False,**kwargs):
        """
        Returns a function returning an image derived from the prompt and
        the initial image + mask.  Return value depends on the seed at
        the time you call it.  kwargs are 'init_latent' and 'strength'
        """
        
        sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
        )

        mask_image = mask_image[0][0].unsqueeze(0).repeat(4,1,1).unsqueeze(0)
        mask_image = repeat(mask_image, '1 ... -> b ...', b=1)

        with torch.autocast("cuda"):
            self.init_latent = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(init_image)
            ) # move to latent space

        t_enc   = int(strength * steps)
        uc, c   = conditioning

        print(f">> target t_enc is {t_enc} steps")

        @torch.no_grad()
        def make_image(x_T):
            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                self.init_latent,
                torch.tensor([t_enc]).to(self.model.device),
                noise=x_T
            )

            # to replace masked area with latent noise, weighted by inpaint_replace strength
            if inpaint_replace > 0.0:
                print(f'>> inpaint will replace what was under the mask with a strength of {inpaint_replace}')
                l_noise = self.get_noise(kwargs['width'],kwargs['height'])
                inverted_mask = 1.0-mask_image  # there will be 1s where the mask is
                masked_region = (1.0-inpaint_replace) * inverted_mask * z_enc + inpaint_replace * inverted_mask * l_noise
                z_enc   = z_enc * mask_image + masked_region

            # decode it
            samples = sampler.decode(
                z_enc,
                c,
                t_enc,
                img_callback                 = step_callback,
                unconditional_guidance_scale = cfg_scale,
                unconditional_conditioning = uc,
                mask                       = mask_image,
                init_latent                = self.init_latent
            )

            return self.sample_to_image(samples)

        return make_image