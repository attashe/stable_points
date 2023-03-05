import os
import sys
import re
import torch
import skimage
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from omegaconf import OmegaConf
from einops import repeat, rearrange
from main import instantiate_from_config

import cv2
from tqdm import tqdm
from loguru import logger

from ldm.models.diffusion.ddim import DDIMSampler
from pytorch_lightning import seed_everything
from .inpaint_invoke import Inpaint as InpaintInvoke
from .inpaint_invoke import downsampling
from .ddim import DDIMSampler as DDIMSamplerInvoke


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

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

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch

def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    
    seed_everything(seed)
    prng = np.random.RandomState(seed)
    
    start_code = prng.randn(num_samples, 4, h//8, w//8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        with torch.autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h//8, w//8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond={"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h//8, w//8]
            samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim+1.0)/2.0,
                                 min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0,2,3,1)
            result = result*255

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]
    
    return result

def load_model(model_path, config_path):
    config = OmegaConf.load(config_path)
    # model = instantiate_from_config(config.model)
    # model.load_state_dict(torch.load(model_path)["state_dict"], strict=False)
    
    model = load_model_from_config(config, model_path)
    sampler = DDIMSampler(model)
    return model, sampler


class Inpainter:
    def __init__(self, model_path, config_path, device="cuda"):
        self.model, self.sampler = load_model(model_path, config_path)
        
        self.device = device
        self.model = self.model.to(device)
        
    def inpaint(self, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
        return inpaint(self.sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples, w, h)


def make_batch(image, mask, device):
    if isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))
    image = np.array(image)
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    if isinstance(mask, str):
        mask = np.array(Image.open(mask).convert("L"))
    mask = np.array(mask)
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


def img2img_inpainter(sampler, model, image, mask, prompt, seed, scale, ddim_steps, strength=0.999, ddim_eta=0.0, device='cuda', w=512, h=512):
    seed_everything(seed)
    assert prompt is not None
    batch_size = 1
    batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=batch_size)
    
    data = [batch_size * [prompt]]
    
    # model.cond_stage_model.encode
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(batch['image']))  # move to latent space
    init_latent = repeat(init_latent, "1 ... -> b ...", b=batch_size)

    # mask = load_mask(image['mask'], Height, Width, init_latent.shape[2], init_latent.shape[3], True).to(device)
    mask = batch['mask'][0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)
    mask = repeat(mask, '1 ... -> b ...', b=batch_size)

    if strength == 1:
        print("strength should be less than 1, setting it to 0.999")
        strength = 0.999
    assert 0.0 <= strength < 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")
    
    all_samples = []
    seeds = ""
    with torch.no_grad():
        with torch.autocast():
            for prompts in tqdm(data, desc="data"):
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                
                c = model.get_learned_conditioning(prompts)

                # encode (scaled latent)
                # z_enc = sampler.stochastic_encode(
                #     init_latent, torch.tensor([t_enc] * batch_size).to(device),
                #     seed, ddim_eta, ddim_steps)
                try:
                    sampler.make_schedule(ddim_num_steps=ddim_steps,  ddim_eta=ddim_eta, verbose=False)
                except Exception:
                    sampler.make_schedule(ddim_num_steps=ddim_steps+1, ddim_eta=ddim_eta, verbose=False)
                
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))

                # decode it
                samples_ddim = sampler.sample(
                    t_enc,
                    c,
                    z_enc,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    mask=mask,
                    x_T=init_latent,
                    sampler=sampler,
                )

                print("saving images")
                for i in range(batch_size):
                    x_samples_ddim = model.decode_first_stage(samples_ddim[i].unsqueeze(0))
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    all_samples.append(x_sample.to("cpu"))
                    x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")

                    seeds += str(seed) + ","
                    seed += 1
                    base_count += 1
    
    torch.cuda.empty_cache()
        
    return Image.fromarray(x_sample.astype(np.uint8))


# shows how the prompt is tokenized
# usually tokens have '</w>' to indicate end-of-word,
# but for readability it has been replaced with ' '
def log_tokenization(text, model, log=False, weight=1):
    if not log:
        return
    tokens    = model.cond_stage_model.tokenizer._tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)
    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', ' ')
        # alternate color
        s = (usedTokens % 6) + 1
        if i < model.cond_stage_model.max_length:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"\x1b[0;3{s};40m{token}"
    print(f"\n>> Tokens ({usedTokens}), Weight ({weight:.2f}):\n{tokenized}\x1b[0m")
    if discarded != "":
        print(
            f">> Tokens Discarded ({totalTokens-usedTokens}):\n{discarded}\x1b[0m"
        )


def split_weighted_subprompts(text, skip_normalize=False)->list:
    """
    grabs all text up to the first occurrence of ':'
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    prompt_parser = re.compile("""
            (?P<prompt>     # capture group for 'prompt'
            (?:\\\:|[^:])+  # match one or more non ':' characters or escaped colons '\:'
            )               # end 'prompt'
            (?:             # non-capture group
            :+              # match one or more ':' characters
            (?P<weight>     # capture group for 'weight'
            -?\d+(?:\.\d+)? # match positive or negative integer or decimal number
            )?              # end weight capture group, make optional
            \s*             # strip spaces after weight
            |               # OR
            $               # else, if no ':' then match end of line
            )               # end non-capture group
            """, re.VERBOSE)
    parsed_prompts = [(match.group("prompt").replace("\\:", ":"), float(
        match.group("weight") or 1)) for match in re.finditer(prompt_parser, text)]
    if skip_normalize:
        return parsed_prompts
    weight_sum = sum(map(lambda x: x[1], parsed_prompts))
    if weight_sum == 0:
        print(
            "Warning: Subprompt weights add up to zero. Discarding and using even weights instead.")
        equal_weight = 1 / max(len(parsed_prompts), 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]


def get_uc_and_c(prompt, model, log_tokens=False, skip_normalize=False):
    # Extract Unconditioned Words From Prompt
    unconditioned_words = ''
    unconditional_regex = r'\[(.*?)\]'
    unconditionals = re.findall(unconditional_regex, prompt)

    if len(unconditionals) > 0:
        unconditioned_words = ' '.join(unconditionals)

        # Remove Unconditioned Words From Prompt
        unconditional_regex_compile = re.compile(unconditional_regex)
        clean_prompt = unconditional_regex_compile.sub(' ', prompt)
        prompt = re.sub(' +', ' ', clean_prompt)

    uc = model.get_learned_conditioning([unconditioned_words])

    # get weighted sub-prompts
    weighted_subprompts = split_weighted_subprompts(
        prompt, skip_normalize
    )

    if len(weighted_subprompts) > 1:
        # i dont know if this is correct.. but it works
        c = torch.zeros_like(uc)
        # normalize each "sub prompt" and add it
        for subprompt, weight in weighted_subprompts:
            log_tokenization(subprompt, model, log_tokens, weight)
            c = torch.add(
                c,
                model.get_learned_conditioning([subprompt]),
                alpha=weight,
            )
    else:   # just standard 1 prompt
        log_tokenization(prompt, model, log_tokens, 1)
        c = model.get_learned_conditioning([prompt])
        uc = model.get_learned_conditioning([unconditioned_words])
    return (uc, c)

class cfg_example:
    
    def __init__(self):
        mconfig             = OmegaConf.load(conf)
        self.model_name     = model
        self.height         = None
        self.width          = None
        self.model_cache    = None
        self.iterations     = 1
        self.steps          = 50
        self.cfg_scale      = 7.5
        self.sampler_name   = sampler_name
        self.ddim_eta       = 0.0    # same seed always produces same image
        self.precision      = precision
        self.strength       = 0.75
        self.seamless       = False
        self.hires_fix      = False
        self.embedding_path = embedding_path
        self.model          = None     # empty for now
        self.model_hash     = None
        self.sampler        = None
        self.device         = None
        self.session_peakmem = None
        self.generators     = {}
        self.base_generator = None
        self.seed           = None
        self.gfpgan = gfpgan
        self.codeformer = codeformer
        self.esrgan = esrgan
        self.free_gpu_mem = free_gpu_mem
        self.size_matters = True  # used to warn once about large image sizes and VRAM


class InitImageResizer():
    """Simple class to create resized copies of an Image while preserving the aspect ratio."""
    def __init__(self,Image):
        self.image = Image

    def resize(self,width=None,height=None) -> Image:
        """
        Return a copy of the image resized to fit within
        a box width x height. The aspect ratio is 
        maintained. If neither width nor height are provided, 
        then returns a copy of the original image. If one or the other is
        provided, then the other will be calculated from the
        aspect ratio.
        Everything is floored to the nearest multiple of 64 so
        that it can be passed to img2img()
        """
        im    = self.image
        
        ar = im.width/float(im.height)

        # Infer missing values from aspect ratio
        if not(width or height): # both missing
            width  = im.width
            height = im.height
        elif not height:           # height missing
            height = int(width/ar)
        elif not width:            # width missing
            width  = int(height*ar)

        # rw and rh are the resizing width and height for the image
        # they maintain the aspect ratio, but may not completelyl fill up
        # the requested destination size
        (rw,rh) = (width,int(width/ar)) if im.width>=im.height else (int(height*ar),height)

        #round everything to multiples of 64
        width,height,rw,rh = map(
            lambda x: x-x%64, (width,height,rw,rh)
        )

        # no resize necessary, but return a copy
        if im.width == width and im.height == height:
            return im.copy()
        
        # otherwise resize the original image so that it fits inside the bounding box
        resized_image = self.image.resize((rw,rh),resample=Image.Resampling.LANCZOS)
        return resized_image


class InpainterStandart:
    def __init__(self, model_path, config_path, device="cuda"):
        self.model, self.sampler = load_model(model_path, config_path)
        
        self.sampler = DDIMSamplerInvoke(self.model, schedule='linear', device=device)
        
        self.device = device
        self.model = self.model.to(device)
        self.skip_normalize = False
        self.log_tokenization = False
        self.inpaint_replace = True
        
        self.inpainter = InpaintInvoke(self.model, 'half')
        
    def inpaint(self, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512, fit=False):
        # batch = make_batch_sd(image, mask, txt=prompt, device=self.device, num_samples=1)
        self.width = w
        self.height = h
        
        uc, c = get_uc_and_c(
                prompt, model =self.model,
                skip_normalize=self.skip_normalize,
                log_tokens    =self.log_tokenization
            )
        
        init_image,mask_image = self._make_images(
                image,
                mask,
                w,
                h,
                fit=fit,
        )
        
        ddim_eta = 0.05
        iterations = 1
        strength = 0.999
        
        results = self.inpainter.generate(
            prompt=prompt,
            iterations=iterations,
            seed=seed,
            sampler=self.sampler,
            steps=ddim_steps,
            cfg_scale=scale,
            conditioning=(uc, c),
            ddim_eta=ddim_eta,
            image_callback=None,  # called after the final image is generated
            step_callback=None,   # called after each intermediate image is generated
            width=w,
            height=h,
            # init_img=init_img,        # embiggen needs to manipulate from the unmodified init_img
            init_image=init_image,      # notice that init_image is different from init_img
            mask_image=mask_image,
            strength=strength,
            # threshold=threshold,
            # perlin=perlin,
            # embiggen=embiggen,
            # embiggen_tiles=embiggen_tiles,
            inpaint_replace=self.inpaint_replace,
            # prompt=prompt, sampler=self.sampler, steps=ddim_steps, cfg_scale=scale,
            # ddim_eta=0.0, init_image=image, strength=0.999,
            # inpaint_replace=False,
        )
        
        sample, seed = results[0]
        # return img2img_inpainter(self.sampler, self.model, image, mask, prompt, seed, scale, ddim_steps)
        
        return [sample,]
    
    def _make_images(
            self,
            img,
            mask,
            width,
            height,
            fit=False,
    ):
        init_image      = None
        init_mask       = None
        if not img:
            return None, None

        image = self._load_img(
            img,
            width,
            height,
        )

        if image.width < self.width and image.height < self.height:
            print(f'>> WARNING: img2img and inpainting may produce unexpected results with initial images smaller than {self.width}x{self.height} in both dimensions')

        # if image has a transparent area and no mask was provided, then try to generate mask
        # if self._has_transparency(image):
        #     self._transparency_check_and_warning(image, mask)
        #     # this returns a torch tensor
        #     init_mask = self._create_init_mask(image, width, height, fit=fit)
            
        if (image.width * image.height) > (self.width * self.height) and self.size_matters:
            print(">> This input is larger than your defaults. If you run out of memory, please use a smaller image.")
            self.size_matters = False

        init_image   = self._create_init_image(image,width,height,fit=fit)                   # this returns a torch tensor

        if mask:
            mask_image = self._load_img(
                mask, width, height)  # this returns an Image
            mask_image = ImageOps.invert(mask_image)
            init_mask = self._create_init_mask(mask_image,width,height,fit=fit)

        return init_image, init_mask

    def correct_colors(self,
                       image_list,
                       reference_image_path,
                       image_callback = None):
        reference_image = Image.open(reference_image_path)
        correction_target = cv2.cvtColor(np.asarray(reference_image),
                                         cv2.COLOR_RGB2LAB)
        for r in image_list:
            image, seed = r
            image = cv2.cvtColor(np.asarray(image),
                                 cv2.COLOR_RGB2LAB)
            image = skimage.exposure.match_histograms(image,
                                                      correction_target,
                                                      channel_axis=2)
            image = Image.fromarray(
                cv2.cvtColor(image, cv2.COLOR_LAB2RGB).astype("uint8")
            )
            if image_callback is not None:
                image_callback(image, seed)
            else:
                r[0] = image
                
    def _load_img(self, img, width, height)->Image:
        if isinstance(img, Image.Image):
            image = img
            print(
                f'>> using provided input image of size {image.width}x{image.height}'
            )
        elif isinstance(img, str):
            assert os.path.exists(img), f'>> {img}: File not found'

            image = Image.open(img)
            print(
                f'>> loaded input image of size {image.width}x{image.height} from {img}'
            )
        else:
            image = Image.open(img)
            print(
                f'>> loaded input image of size {image.width}x{image.height}'
            )
        image = ImageOps.exif_transpose(image)
        return image

    def _create_init_image(self, image, width, height, fit=True):
        image = image.convert('RGB')
        if fit:
            image = self._fit_image(image, (width, height))
        else:
            image = self._squeeze_image(image)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = 2.0 * image - 1.0
        return image.to(self.device)

    def _create_init_mask(self, image, width, height, fit=True):
        # convert into a black/white mask
        # image = self._image_to_mask(image)
        image = image.convert('RGB')

        # now we adjust the size
        if fit:
            image = self._fit_image(image, (width, height))
        else:
            image = self._squeeze_image(image)
        image = image.resize((image.width//downsampling, image.height //
                              downsampling), resample=Image.Resampling.NEAREST)
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image.to(self.device)

    # The mask is expected to have the region to be inpainted
    # with alpha transparency. It converts it into a black/white
    # image with the transparent part black.
    def _image_to_mask(self, mask_image, invert=False) -> Image:
        # Obtain the mask from the transparency channel
        mask = Image.new(mode="L", size=mask_image.size, color=255)
        mask.putdata(mask_image.getdata(band=3))
        if invert:
            mask = ImageOps.invert(mask)
        return mask

    def _has_transparency(self, image):
        if image.info.get("transparency", None) is not None:
            return True
        if image.mode == "P":
            transparent = image.info.get("transparency", -1)
            for _, index in image.getcolors():
                if index == transparent:
                    return True
        elif image.mode == "RGBA":
            extrema = image.getextrema()
            if extrema[3][0] < 255:
                return True
        return False

    def _check_for_erasure(self, image):
        width, height = image.size
        pixdata = image.load()
        colored = 0
        for y in range(height):
            for x in range(width):
                if pixdata[x, y][3] == 0:
                    r, g, b, _ = pixdata[x, y]
                    if (r, g, b) != (0, 0, 0) and \
                       (r, g, b) != (255, 255, 255):
                        colored += 1
        return colored == 0

    def _transparency_check_and_warning(self,image, mask):
        if not mask:
            print(
                '>> Initial image has transparent areas. Will inpaint in these regions.')
            if self._check_for_erasure(image):
                print(
                    '>> WARNING: Colors underneath the transparent region seem to have been erased.\n',
                    '>>          Inpainting will be suboptimal. Please preserve the colors when making\n',
                    '>>          a transparency mask, or provide mask explicitly using --init_mask (-M).'
                )

    def _squeeze_image(self, image):
        x, y, resize_needed = self._resolution_check(image.width, image.height)
        if resize_needed:
            return InitImageResizer(image).resize(x, y)
        return image

    def _fit_image(self, image, max_dimensions):
        w, h = max_dimensions
        print(
            f'>> image will be resized to fit inside a box {w}x{h} in size.'
        )
        if image.width > image.height:
            h = None   # by setting h to none, we tell InitImageResizer to fit into the width and calculate height
        elif image.height > image.width:
            w = None   # ditto for w
        else:
            pass
        # note that InitImageResizer does the multiple of 64 truncation internally
        image = InitImageResizer(image).resize(w, h)
        print(
            f'>> after adjusting image dimensions to be multiples of 64, init image is {image.width}x{image.height}'
        )
        return image

    def _resolution_check(self, width, height, log=False):
        resize_needed = False
        w, h = map(
            lambda x: x - x % 64, (width, height)
        )  # resize to integer multiple of 64
        if h != height or w != width:
            if log:
                print(
                    f'>> Provided width and height must be multiples of 64. Auto-resizing to {w}x{h}'
                )
            height = h
            width = w
            resize_needed = True

        return width, height, resize_needed