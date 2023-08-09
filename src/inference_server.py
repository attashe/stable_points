"""
1. Initialize model
2. Load weights
3. Serve http port
"""
import os
import torch
os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "1"
os.environ['XDG_CACHE_HOME'] = 'K:/Weights/'
import sys
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream, style_transfer, super_resolution, inpainting
import torch.nn.functional as F
import random
import torchvision.transforms as T
import numpy as np
import requests
from PIL import Image
import torch
import re

import configargparse
from flask import Flask, render_template, send_file, send_from_directory, request, jsonify

import json
import numpy as np
import cv2
from PIL import Image

from typing import Optional, Dict

app = Flask(__name__, static_folder='./static')
app.config['SECRET_KEY'] = 'secret!'


class Model:
    
    def __init__(self):
        self.device = 'cuda'
        self.if_I = IFStageI('IF-I-XL-v1.0', device=self.device, cache_dir='K:/Weights/')
        self.if_II = IFStageII('IF-II-L-v1.0', device=self.device, cache_dir='K:/Weights/')
        # self.if_III = StableStageIII('stable-diffusion-x4-upscaler', device=self.device, cache_dir='K:/Weights/')

        self.t5 = T5Embedder(device=self.device, cache_dir='K:/Weights/', use_offload_folder='K:/Weights/offload/')

    def txt2img(self, prompt: str, negative_prompt: str, count=1, seed=42, IF_I_kwargs={}, IF_II_kwargs={}):
        # prompt = "a photo of women with plate 'I ❤️ dicks'"
        # negative_prompt = 'poor quality, ugly, bad face'

        result = dream(
            t5=self.t5, if_I=self.if_I, if_II=self.if_II, #if_III=if_III,
            prompt=[prompt]*count,
            negative_prompt=[negative_prompt]*count,
            seed=seed,
            if_I_kwargs=IF_I_kwargs,
            if_II_kwargs=IF_II_kwargs,
            disable_watermark=True,
        )
        
        return result['II'][0]
        
class Context:
    model: Optional[Model] = None
    

def load_model():
    return Model()


def run_txt2img(args):
    if Context.model is None:
        Context.model = load_model()
        
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    }
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    }
    
    print(f'Txt2Img with parameters:\n args: {args}')
    
    img = Context.model.txt2img(
        args['prompt'],
        args['negative_prompt'],
        args['count'],
        args['seed'],
        IF_I_kwargs=if_I_kwargs, IF_II_kwargs=if_II_kwargs
    )
    
    # save_img(img)
    
    return img


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5005)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    return args


@app.route('/post_txt2img', methods=['POST'])
def txt2img():
    json_data = json.loads(request.files['json'].read())
    
    img = run_txt2img(json_data)
    # img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    
    img.save('example.png')
    
    _, res = cv2.imencode('.png', np.array(img))
    
    return res.tobytes()


@app.route('/post_img2img', methods=['POST'])
def get_image():
    print(request.headers)
    # print(request.data)
    print(json.loads(request.files['json'].read()))
    # print('JSON: ', request.json)
    nparr = np.frombuffer(request.files['media'].read(), dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imshow('Image', img)
    cv2.waitKey()
    
    if request.args.get('type') == '1':
       filename = '../test_imgs/0001.png'
    else:
       filename = '../test_imgs/0001.png'
    # return send_file(filename, mimetype='image/gif')
    _, res = cv2.imencode('.png', np.array(img))

    return res.tostring()

def main():
    app.run(port=5005, debug=False)


if __name__ == "__main__":
    args = parse_args()
    main()
