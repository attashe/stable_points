import os
import numpy as np
import torch

from itertools import islice
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from torch import autocast
from contextlib import nullcontext
from loguru import logger

from einops import rearrange
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


torch.set_grad_enabled(False)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class config():
    def __init__(self):    
        self.outdir = 'outputs/txt2img-samples'
        self.ddim_steps = 50
        self.sampler = 'ddim'
        
        self.seed = 42
        # self.config = 'configs/stable-diffusion/v2-inference.yaml'
        # self.ckpt = 'J:/Weights/sd-2.0/512-base-ema.ckpt'
        self.config = 'configs/stable-diffusion/v2-inference-v.yaml'
        self.ckpt = 'G:/Weights/sd-2.0/768-v-ema.ckpt'
        self.precision = 'autocast'
        self.n_rows = 0
        self.fixed_code = False
        self.from_file = False
        self.C = 4
        self.H = 768  # 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024
        self.W = 768
        self.f = 8
        self.n_samples = 2
        self.n_iter = 1
        self.scale = 11.0
        self.ddim_eta = 0.0
        self.skip_save = False
        self.skip_grid = True


class SDModel:
    
    def __init__(self, model_path, cfg_path) -> None:
        logger.info('Start loading model')
        self.config = OmegaConf.load(str(cfg_path))
        self.model = load_model_from_config(self.config, str(model_path))
        logger.info('Model was loaded')


class Txt2ImgInference:
    
    def __init__(self, model: SDModel = None):
        self.opt = config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.model
        logger.info(f'Device is {self.device} and model device is {self.model.device}')
        self.model.to(self.device)
        
    def predict(self, text, seed=42):
        seed_everything(seed)
        return self.generate(text)
        
    def generate(self, prompt, width, height, sampler, steps, scale):
        device = self.device
        opt = self.opt
        
        if sampler == 'plms':
            sampler = PLMSSampler(self.model)
        elif sampler == 'dpm':
            sampler = DPMSolverSampler(self.model)
        elif sampler == 'ddim':
            sampler = DDIMSampler(self.model)
        else:
            raise Exception('Unknown sampler type')
        images = []

        batch_size = 1

        assert prompt is not None
        data = [batch_size * [prompt]]

        sample_count = 0

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, height // opt.f, width // opt.f], device=device)

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    #tic = time.time()
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [opt.C, height // opt.f, width // opt.f]
                            samples_ddim, _ = sampler.sample(S=steps,
                                                            conditioning=c,
                                                            batch_size=batch_size,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                            x_samples = self.model.decode_first_stage(samples_ddim)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                images +=[Image.fromarray(x_sample.astype(np.uint8))]
                                sample_count += 1
                                
                            all_samples.append(x_samples)

                    return images


# ===================================================
# Loader from automatic web-gui

def select_checkpoint():
    model_checkpoint = shared.opts.sd_model_checkpoint
        
    checkpoint_info = checkpoint_alisases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        print("No checkpoints found. When searching for checkpoints, looked at:", file=sys.stderr)
        if shared.cmd_opts.ckpt is not None:
            print(f" - file {os.path.abspath(shared.cmd_opts.ckpt)}", file=sys.stderr)
        print(f" - directory {model_path}", file=sys.stderr)
        if shared.cmd_opts.ckpt_dir is not None:
            print(f" - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}", file=sys.stderr)
        print("Can't run without a checkpoint. Find and place a .ckpt file into any of those locations. The program will exit.", file=sys.stderr)
        exit(1)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info

import time
class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        end = time.time()
        res = end - self.start
        self.start = end
        return res

class shared:
    no_half = False

def load_model(model_path, cfg_path):
    from src import sd_hijack
    from src import sd_disable_initialization

    print(f"Loading config from: {cfg_path}")

    # if shared.sd_model:
    #     sd_hijack.model_hijack.undo_hijack(shared.sd_model)
    #     shared.sd_model = None
    #     gc.collect()
    #     devices.torch_gc()

    sd_config = OmegaConf.load(cfg_path)
    
    # if should_hijack_inpainting(checkpoint_info):
    #     # Hardcoded config for now...
    #     sd_config.model.target = "ldm.models.diffusion.ddpm.LatentInpaintDiffusion"
    #     sd_config.model.params.conditioning_key = "hybrid"
    #     sd_config.model.params.unet_config.params.in_channels = 9
    #     sd_config.model.params.finetune_keys = None

    if not hasattr(sd_config.model.params, "use_ema"):
        sd_config.model.params.use_ema = False

    # do_inpainting_hijack()

    if shared.no_half:
        sd_config.model.params.unet_config.params.use_fp16 = False

    timer = Timer()

    sd_model = None

    try:
        with sd_disable_initialization.DisableInitialization():
            sd_model = instantiate_from_config(sd_config.model)
    except Exception as e:
        pass

    if sd_model is None:
        print('Failed to create model quickly; will retry using slow method.', file=sys.stderr)
        sd_model = instantiate_from_config(sd_config.model)

    elapsed_create = timer.elapsed()

    load_model_weights(sd_model, checkpoint_info)

    elapsed_load_weights = timer.elapsed()

    sd_model.to(device)

    sd_hijack.model_hijack.hijack(sd_model)

    sd_model.eval()

    elapsed_the_rest = timer.elapsed()

    print(f"Model loaded in {elapsed_create + elapsed_load_weights + elapsed_the_rest:.1f}s ({elapsed_create:.1f}s create model, {elapsed_load_weights:.1f}s load weights).")

    return sd_model