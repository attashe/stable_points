# GUI application for create animation with stable diffusion model

This script create pointcloud from image and apply different transformation to create series of images with camera movement.

## Showcase

WIP interface preview:

![WIP interface previ

https://user-images.githubusercontent.com/8243605/226091205-3267c067-43d5-4034-afc4-c38743785773.mov

ew!](images/interface_preview_01.jpg "Interface and loaded image")



This program uses Stable Diffusion Inpainting model for filling gaps caused by rotation camera. Depthmap is estimated by MiDaS model or LeReS.

For pointcloud rendering using pure pytorch render.

## Installation

1. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

2. Download weights

    TODO: add stable diffusion weights link
    TODO: add depth model weights link
    TODO: add FILM weights

3. Install repositories

    - LeRes
    - MiDaS
    - FILM-pytorch

## Using

1. Prepare init image
2. Choose depthscale
3. Rotate camera
4. Select inpainting mask
5. Set inpainting options
6. Save Keyframe
7. Repeat points 2-6
8. Create animation with frame interpolation and optional upscaling

## Dependencies

1. PyTorch
2. Stable Diffusion
3. MiDaS, LeReS
4. DearPyGUI
5. OpenCV
6. Automatic1111 API

## Examples

1. 3D camera movement

https://user-images.githubusercontent.com/8243605/226091007-2b5d6f7b-c33a-4c4c-b207-73b4e4dc1b43.mov

2. Img2Img with zoom example

https://user-images.githubusercontent.com/8243605/226091187-29aec9ff-e6c8-4dcb-aa23-1d40c0037b66.mov


