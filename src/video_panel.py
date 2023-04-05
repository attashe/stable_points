"""Video processing
Наложение видео сверху с помощью оптического потока. Также необходимо реализовать варианты без его применения.
В идеале - встроить в систему Nodes. Код основывается на Ward DiscoDiffusion from Xsella
"""
import os
import gc
import sys
import argparse
import subprocess

from glob import glob
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import cv2
import torch
import webuiapi
import numpy as np

from loguru import logger


def load_video(filename: str):
    # read video
    cap = cv2.VideoCapture(filename)
    # get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        
        yield frame

def load_img(img, size):
    img = Image.open(img).convert('RGB').resize(size)
    return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...].cuda()

def write_video(filename: str):
    pass


def process_video():
    pass


class OpticalFlowInference:
    DEBUG = False
    
    def __init__(self):
        self.raft_path = 'G:/Python Scripts/RAFT'
        sys.path.append(f'{self.raft_path}/core')
        
        try:
            from raftutils.utils import InputPadder
            from raft import RAFT
            self.InputPadder = InputPadder
        except:
            raise Exception(f'Unable to find RAFT model files in the {self.raft_path} directory')
        
        args2 = argparse.Namespace()
        args2.small = False
        args2.mixed_precision = True

        TAG_CHAR = np.array([202021.25], np.float32)
        
        self.raft_model = torch.nn.DataParallel(RAFT(args2))
        self.raft_model.load_state_dict(torch.load(f'{self.raft_path}/models/raft-things.pth'))
        self.raft_model = self.raft_model.module.cuda().eval()

    def get_flow(self, frame1, frame2, iters=20):
        padder = self.InputPadder(frame1.shape)
        frame1, frame2 = padder.pad(frame1, frame2)
        _, flow12 = self.raft_model(frame1, frame2, iters=iters, test_mode=True)
        flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()

        return flow12
    
    @staticmethod
    def load_cc(path):
        weights = np.load(path)
        weights = np.repeat(weights[...,None],3, axis=2)
        if OpticalFlowInference.DEBUG: print('weight min max mean std', weights.shape, weights.min(), weights.max(), weights.mean(), weights.std())
        return weights

    @staticmethod
    def warp_flow(img, flow):
        h, w = flow.shape[:2]
        flow = flow.copy()
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res
    
class WarpDiffusion:
    
    def __init__(self) -> None:
        self.blend_mode = 'optical_flow'  # optical_flow | linear | etc
        self.blend = 0.5
        self.fps = 15
        
    def prepare_flow(self, video_path):
        pass



def make_frames(video_path, target_folder, extract_nth_frame=1):
    # Make folder, if already exists clean old files
    Path(target_folder).mkdir(exist_ok=True)
    for f in Path(f'{target_folder}').glob('*.jpg'):
        f.unlink()

    vf = f'select=not(mod(n\,{extract_nth_frame}))'
    if os.path.exists(video_path):
        subprocess.run(['ffmpeg', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{target_folder}/%04d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    else: 
        print(f'\nWARNING!\n\nVideo not found: {video_path}.\nPlease check your video path.\n')
    #!ffmpeg -i {video_init_path} -vf {vf} -vsync vfr -q:v 2 -loglevel error -stats {videoFramesFolder}/%04d.jpg


def generate_optical_flow(frames_path: str, flow_path: str, width_height: tuple[int, int],
                          raft_model: OpticalFlowInference, check_consistency=True):
    """Check consistency parameter needs for fix a lot of artifacts"""

    frames = sorted(glob(frames_path+'/*.*'));
    assert len(frames)>=2, f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.'

    Path(flow_path).mkdir(exist_ok=True)
    for f in Path(f'{flow_path}').glob('*.jpg*'):
        f.unlink()

    Path(flow_path).mkdir(exist_ok=True)

    # TODO: write a code for clone side repos and prepare an environment
    cc_path = f'E:/GitHub/stable_points/notebooks/flow_tools/check_consistency.py'

    for frame1, frame2 in tqdm(zip(frames[:-1], frames[1:]), total=len(frames)-1):

        out_flow21_fn = f"{flow_path}/{Path(frame1).name}"

        frame1 = load_img(frame1, width_height)
        frame2 = load_img(frame2, width_height)

        flow21 = raft_model.get_flow(frame2, frame1)
        np.save(out_flow21_fn, flow21.astype('float32'))

        if check_consistency:
            flow12 = raft_model.get_flow(frame1, frame2)
            np.save(out_flow21_fn+'_12', flow12.astype('float32'))
            gc.collect()
            
    if check_consistency:
        fwd = f"{flow_path}/*jpg.npy"
        bwd = f"{flow_path}/*jpg_12.npy"
        
        # !python "{cc_path}" --flow_fwd "{fwd}" --flow_bwd "{bwd}" --output "{flo_fwd_folder}/" --image_output --output_postfix="-21_cc"
        logger.info('Run check consistency script')
        logger.info(f'{cc_path} {fwd} {bwd} {flow_path}')
        logger.info(' '.join([
            'python',
            f'{cc_path}',
            '--flow_fwd',
            f'{fwd}',
            '--flow_bwd',
            f'{bwd}',
            '--output',
            f'{flow_path}',
            '--image_output',
            '--output_postfix="-21_cc"'
        ]))
        subprocess.run([
            'python',
            f'{cc_path}',
            '--flow_fwd',
            f'{fwd}',
            '--flow_bwd',
            f'{bwd}',
            '--output',
            f'{flow_path}',
            '--image_output',
            '--output_postfix=-21_cc'
        ])
        
    del raft_model
    gc.collect()

def get_video_size(filename: str):
    # read video
    cap = cv2.VideoCapture(filename)
    # get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    cap.release()
    
    return (width, height)


class SDconfig:
    seed = 1234
    # prompt = 'Beuatiful park on the Mars'
    # prompt = 'Emma Watson walking in bikini, anime style'
    # prompt = 'Photo portrait of (Emma Watson), volumetric lighting, high detailed'
    prompt = 'Cyberpunk ninja robot with long energy spear, high detailed, volumetric lighting'
    negative_prompt = 'blurry, poor quality, malformed'
    ddim_steps = 20
    scale = 7.0
    strength = 0.45
    use_controlnet = True
    # controlnets = set(['depth'],)
    # controlnets = set(['canny'],)
    controlnets = set(['canny', 'depth'])


def apply_img2img(api, img_pil, width, height, controlnet_init=None):
    controlnets = []
    controlnet_init = img_pil if controlnet_init is None else controlnet_init
    if SDconfig.use_controlnet:
        if 'canny' in SDconfig.controlnets:
            controlnets.append(webuiapi.ControlNetUnit(
                input_image=controlnet_init, module='canny', model='control_canny-fp16 [e3fe7712]', weight=1.0))
        if 'depth' in SDconfig.controlnets:
            controlnets.append(webuiapi.ControlNetUnit(
                input_image=controlnet_init, module='depth', model='control_depth-fp16 [400750f6]', weight=1.0))
            
    logger.debug(controlnets)
    
    img2img_result = api.img2img(prompt=SDconfig.prompt,
            negative_prompt=SDconfig.negative_prompt,
            images=[img_pil], 
            width=width,
            height=height,
            controlnet_units=controlnets,
            sampler_name="Euler a",
            steps=SDconfig.ddim_steps,
            cfg_scale=SDconfig.scale,
            seed=SDconfig.seed,
            eta=1.0,
            denoising_strength=SDconfig.strength,
        )

    return img2img_result.image

def iterate_img2img_orig(img_folder, flow_folder, save_folder,
                         blend, check_consistency=True):
    Path(save_folder).mkdir(exist_ok=True)
    
    # Create automatic1111 API
    api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)
    
    imgs = sorted(list(Path(img_folder).glob('*.jpg')))
    
    prev = None
    weights_path = None
    
    # TODO: Batch processing
    for i in range(len(imgs)):
        img_path = f"{img_folder}/{(i+1):04}.jpg"
        img_pil = Image.open(img_path)
        
        img_arr = np.array(img_pil)
        height, width = img_arr.shape[:2]
        logger.info(f'Start inference with automatic API with next parameters:')
        logger.info(f'{SDconfig.prompt=}, {SDconfig.ddim_steps=}, {SDconfig.scale=}, {SDconfig.strength=}, {SDconfig.use_controlnet=}, {SDconfig.seed=}')
        logger.info(f'size = {width}x{height}')

        img2img_result = apply_img2img(api, img_pil, width, height)
        
        if i == 0:
            img2img_result.save(save_folder + f'/{(i+1):04}.png')
            prev = img2img_result
            continue
        
        flow_idx = f'{i:04}'
        flow1_path = f"{flow_folder}/{flow_idx}.jpg.npy"
        flow21 = np.load(flow1_path)

        img = img2img_result
        prev_arr = np.array(prev.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
        img_arr = np.array(img.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
        
        frame1_warped21 = OpticalFlowInference.warp_flow(prev_arr, flow21)
        
        weights_path = None
        if check_consistency: weights_path = f"{flow_folder}/{flow_idx}.jpg-21_cc.npy"
        print(weights_path)

        if weights_path: 
            forward_weights = OpticalFlowInference.load_cc(weights_path)
            blended_w = img_arr*(1-blend) + blend*(frame1_warped21*forward_weights+img_arr*(1-forward_weights))
        else: blended_w = img_arr*(1-blend) + frame1_warped21*(blend)
        
        blend_pil = Image.fromarray(blended_w.astype('uint8'))

        blend_pil.save(save_folder + f'/{(i+1):04}.png')
        prev = blend_pil

def iterate_img2img(img_folder, flow_folder, save_folder,
                    blend, check_consistency=True):
    Path(save_folder).mkdir(exist_ok=True)
    
    # Create automatic1111 API
    api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)
    
    imgs = sorted(list(Path(img_folder).glob('*.jpg')))
    
    prev = None
    weights_path = None

    # TODO: Batch processing
    for i in range(len(imgs)):
        img_path = f"{img_folder}/{(i+1):04}.jpg"
        img_pil = Image.open(img_path)
        
        img_arr = np.array(img_pil)
        height, width = img_arr.shape[:2]
        logger.info(f'Start inference with automatic API with next parameters:')
        logger.info(f'{SDconfig.prompt=}, {SDconfig.ddim_steps=}, {SDconfig.scale=}, {SDconfig.strength=}, {SDconfig.use_controlnet=}, {SDconfig.seed=}')
        logger.info(f'size = {width}x{height}')
        
        if i == 0:
            img2img_result = apply_img2img(api, img_pil, width, height)
            
            img2img_result.save(save_folder + f'/{(i+1):04}.png')
            prev = img2img_result
            continue
        
        flow_idx = f'{i:04}'
        flow1_path = f"{flow_folder}/{flow_idx}.jpg.npy"
        flow21 = np.load(flow1_path)
        
    #     img = img2img_result.image
        prev_arr = np.array(prev.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
        img_arr = np.array(img_pil.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
        
        frame1_warped21 = OpticalFlowInference.warp_flow(prev_arr, flow21)
        
        weights_path = None
        if check_consistency: weights_path = f"{flow_folder}/{flow_idx}.jpg-21_cc.npy"
        print(weights_path)

        if weights_path: 
            forward_weights = OpticalFlowInference.load_cc(weights_path)
            blended_w = img_arr*(1-blend) + blend*(frame1_warped21*forward_weights+img_arr*(1-forward_weights))
        else: blended_w = img_arr*(1-blend) + frame1_warped21*(blend)

        blend_pil = Image.fromarray(blended_w.astype('uint8'))
        
        img2img_result = apply_img2img(api, blend_pil, width, height)
        blend_pil = img2img_result
        blend_pil.save(save_folder + f'/{(i+1):04}.png')
        prev = blend_pil

def iterate_img2img_v2(img_folder, flow_folder, save_folder,
                    blend, check_consistency=True):
    Path(save_folder).mkdir(exist_ok=True)
    
    # Create automatic1111 API
    api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)
    
    imgs = sorted(list(Path(img_folder).glob('*.jpg')))
    
    prev = None
    weights_path = None

    # TODO: Batch processing
    for i in range(len(imgs)):
        img_path = f"{img_folder}/{(i+1):04}.jpg"
        img_pil = Image.open(img_path)
        
        img_arr = np.array(img_pil)
        height, width = img_arr.shape[:2]
        logger.info(f'Start inference with automatic API with next parameters:')
        logger.info(f'{SDconfig.prompt=}, {SDconfig.ddim_steps=}, {SDconfig.scale=}, {SDconfig.strength=}, {SDconfig.use_controlnet=}, {SDconfig.seed=}')
        logger.info(f'size = {width}x{height}')
        
        if i == 0:
            img2img_result = apply_img2img(api, img_pil, width, height)
            
            img2img_result.save(save_folder + f'/{(i+1):04}.png')
            prev = img2img_result
            continue
        
        flow_idx = f'{i:04}'
        flow1_path = f"{flow_folder}/{flow_idx}.jpg.npy"
        flow21 = np.load(flow1_path)
        
    #     img = img2img_result.image
        prev_arr = np.array(prev.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
        img_arr = np.array(img_pil.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
        
        frame1_warped21 = OpticalFlowInference.warp_flow(prev_arr, flow21)
        
        weights_path = None
        if check_consistency: weights_path = f"{flow_folder}/{flow_idx}.jpg-21_cc.npy"
        print(weights_path)

        if weights_path: 
            forward_weights = OpticalFlowInference.load_cc(weights_path)
            blended_w = frame1_warped21*forward_weights+img_arr*(1-forward_weights)
        else: blended_w = frame1_warped21

        blend_pil = Image.fromarray(blended_w.astype('uint8'))
        
        img2img_result = apply_img2img(api, blend_pil, width, height)
        # img2img_result = np.array(img2img_result)
        
        if weights_path:
            img2img_result = np.array(img2img_result.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
            blended_w = img2img_result*(1-blend) + blend*(blended_w*forward_weights+img2img_result*(1-forward_weights))
            blend_pil = Image.fromarray(blended_w.astype('uint8'))
        else: blend_pil = img2img_result
        
        blend_pil.save(save_folder + f'/{(i+1):04}.png')
        prev = blend_pil

def iterate_img2img_v3(img_folder, flow_folder, save_folder,
                    blend, check_consistency=True):
    Path(save_folder).mkdir(exist_ok=True)
    
    # Create automatic1111 API
    api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)
    
    imgs = sorted(list(Path(img_folder).glob('*.jpg')))
    
    prev = None
    weights_path = None

    # TODO: Batch processing
    for i in range(len(imgs)):
        img_path = f"{img_folder}/{(i+1):04}.jpg"
        img_pil = Image.open(img_path)
        
        img_arr = np.array(img_pil)
        height, width = img_arr.shape[:2]
        logger.info(f'Start inference with automatic API with next parameters:')
        logger.info(f'{SDconfig.prompt=}, {SDconfig.ddim_steps=}, {SDconfig.scale=}, {SDconfig.strength=}, {SDconfig.use_controlnet=}, {SDconfig.seed=}')
        logger.info(f'size = {width}x{height}')
        
        if i == 0:
            img2img_result = apply_img2img(api, img_pil, width, height)
            
            img2img_result.save(save_folder + f'/{(i+1):04}.png')
            prev = img2img_result
            continue
        
        flow_idx = f'{i:04}'
        flow1_path = f"{flow_folder}/{flow_idx}.jpg.npy"
        flow21 = np.load(flow1_path)
        
    #     img = img2img_result.image
        prev_arr = np.array(prev.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
        img_arr = np.array(img_pil.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
        
        frame1_warped21 = OpticalFlowInference.warp_flow(prev_arr, flow21)
        
        weights_path = None
        if check_consistency: weights_path = f"{flow_folder}/{flow_idx}.jpg-21_cc.npy"
        print(weights_path)

        if weights_path: 
            forward_weights = OpticalFlowInference.load_cc(weights_path)
            blended_w = img_arr*(1-blend) + blend*(frame1_warped21*forward_weights+img_arr*(1-forward_weights))
        else: blended_w = img_arr*(1-blend) + frame1_warped21*(blend)

        blend_pil = Image.fromarray(blended_w.astype('uint8'))
        
        img2img_result = apply_img2img(api, blend_pil, width, height, controlnet_init=Image.fromarray(img_arr))
        blend_pil = img2img_result
        blend_pil.save(save_folder + f'/{(i+1):04}.png')
        prev = blend_pil
        
def iterate_img2img_v4(img_folder, flow_folder, save_folder,
                    blend, check_consistency=True):
    Path(save_folder).mkdir(exist_ok=True)
    
    # Create automatic1111 API
    api = webuiapi.WebUIApi(host='127.0.0.1', port=7860)
    
    imgs = sorted(list(Path(img_folder).glob('*.jpg')))
    
    prev = None
    weights_path = None
    time_arr = None

    # TODO: Batch processing
    for i in range(len(imgs)):
        img_path = f"{img_folder}/{(i+1):04}.jpg"
        img_pil = Image.open(img_path)
        
        img_arr = np.array(img_pil)
        height, width = img_arr.shape[:2]
        logger.info(f'Start inference with automatic API with next parameters:')
        logger.info(f'{SDconfig.prompt=}, {SDconfig.ddim_steps=}, {SDconfig.scale=}, {SDconfig.strength=}, {SDconfig.use_controlnet=}, {SDconfig.seed=}')
        logger.info(f'size = {width}x{height}')
        
        if i == 0:
            _strength = SDconfig.strength
            SDconfig.strength = 0.6
            img2img_result = apply_img2img(api, img_pil, width, height)
            SDconfig.strength = _strength
            
            img2img_result.save(save_folder + f'/{(i+1):04}.png')
            prev = img2img_result
            continue
        
        flow_idx = f'{i:04}'
        flow1_path = f"{flow_folder}/{flow_idx}.jpg.npy"
        flow21 = np.load(flow1_path)
        
    #     img = img2img_result.image
        prev_arr = np.array(prev.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
        img_arr = np.array(img_pil.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
        
        time_arr = np.ones_like(img_arr) if time_arr is None else time_arr + 1
        
        frame1_warped21 = OpticalFlowInference.warp_flow(prev_arr, flow21)
        time_arr = OpticalFlowInference.warp_flow(time_arr, flow21)
        
        weights_path = None
        if check_consistency: weights_path = f"{flow_folder}/{flow_idx}.jpg-21_cc.npy"
        print(weights_path)

        if weights_path: 
            forward_weights = OpticalFlowInference.load_cc(weights_path)
            
            time_arr = time_arr * (1-forward_weights)
            time_arr = np.clip(time_arr, 0, 10)
            blend_upd = blend * (time_arr / 10)
            blend_upd = np.clip(blend_upd, 0.2, 1)
            
            logger.info(f'Blend mean : {blend_upd.mean()}')
            
            blended_w = img_arr*(1-blend_upd) + blend_upd*(frame1_warped21*forward_weights+img_arr*(1-forward_weights))
        else: blended_w = img_arr*(1-blend) + frame1_warped21*(blend)

        blend_pil = Image.fromarray(blended_w.astype('uint8'))
        
        img2img_result = apply_img2img(api, blend_pil, width, height, controlnet_init=Image.fromarray(img_arr))
        blend_pil = img2img_result
        blend_pil.save(save_folder + f'/{(i+1):04}.png')
        prev = blend_pil

def create_video_file(frames_path, save_path, fps=12, init_frame=1):
    image_path = f"{frames_path}/%04d.png"
    
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec',
        'png',
        '-r',
        str(fps),
        '-start_number',
        str(init_frame),
        '-i',
        image_path,
        # '-frames:v',
        # str(last_frame+1),
        '-c:v',
        'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt',
        'yuv420p',
        '-crf',
        '17',
        '-preset',
        'veryslow',
        save_path
    ]

    process = subprocess.Popen(cmd, cwd=f'.', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
    else:
        print("The video is ready and saved to the images folder")

def main():
    # Parameters
    video = 'E:/GitHub/stable_points/test_clips/vlad_hd.mp4'
    base_path = Path('./output/video_proc_10/')
    base_path.mkdir(exist_ok=True)
    save_frames_folder = str(base_path / 'frames')
    save_flow_folder = str(base_path / 'flow')
    save_res_folder = str(base_path / 'result')
    
    extract_nth_frame = 1
    check_consistency = True
    fps = 20
    blend = 0.5
    print(f"Exporting Video Frames (1 every {extract_nth_frame})...")
    
    # Run the code
    width_height = get_video_size(video)
    raft_infer = OpticalFlowInference()
    
    # make_frames(video, save_frames_folder, extract_nth_frame=extract_nth_frame)
    
    # generate_optical_flow(save_frames_folder, save_flow_folder, width_height,
    #                       raft_infer, check_consistency)
    # TODO: Save all params to config for history
    iterate_img2img_v4(save_frames_folder, save_flow_folder, save_res_folder,
                    blend=blend, check_consistency=check_consistency)
    create_video_file(save_res_folder, 'output_vlad_hd_v4_control_canny_depth_skip_1_b05_str045.mp4', fps=fps)
    # video_enhance


if __name__ == "__main__":
    main()