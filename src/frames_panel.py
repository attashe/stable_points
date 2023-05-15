import os
import bisect
import cv2
import numpy as np

import itertools
from tqdm import tqdm

import torch
import dearpygui.dearpygui as dpg

from loguru import logger

from context import Context

class Frame:
    
    def __init__(self, name: str, steps: int) -> None:
        self.image = load_image(name)
        self.steps = steps
    
    @staticmethod
    def interpolate(frame1, frame2, steps):
        # 1. init interpolator
        # 2. load frame1 and frame2 to tensor
        frame1_torch = None
        frame2_torch = None
        
        # 3. Make intermediate frames
        ...


class FramesContainer:
    def __init__(self, frames=[]):
        self.queue: Frame = frames

    def add_frane(self, frame):
        self.queue.append(frame)

    def remove_frame(self, frame):
        self.queue.remove(frame)

    def move_frame_forward(self, frame):
        index = self.queue.index(frame)
        if index < len(self.queue) - 1:
            self.queue[index], self.queue[index + 1] = self.queue[index + 1], self.queue[index]

    def move_frame_backward(self, frame):
        index = self.queue.index(frame)
        if index > 0:
            self.queue[index], self.queue[index - 1] = self.queue[index - 1], self.queue[index]

    def insert_frame(self, frame, position):
        self.queue.insert(position, frame)


def pad_batch(batch, align):
    height, width = batch.shape[1:3]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [height_to_pad >> 1, width_to_pad >> 1, height + (height_to_pad >> 1), width + (width_to_pad >> 1)]
    batch = np.pad(batch, ((0, 0), (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                           (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)), (0, 0)), mode='constant')
    return batch, crop_region


def load_image(path, align=64):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
    image_batch, crop_region = pad_batch(np.expand_dims(image, axis=0), align)
    return image_batch, crop_region


class FrameInterpolationModel:
    
    def __init__(self, model_path, half=True, gpu=True) -> None:
        self.device = torch.device('cuda') if gpu else torch.device('cpu')
        self.gpu = gpu
        self.half = half
        
        self._load_model(model_path)
        
    def infer(self, img1, img2, inter_frames):
        img_batch_1, crop_region_1 = load_image(img1)
        img_batch_2, crop_region_2 = load_image(img2)

        img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
        img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)
        
        results = [
            img_batch_1,
            img_batch_2
        ]
        
        idxes = [0, inter_frames + 1]
        remains = list(range(1, inter_frames + 1))

        splits = torch.linspace(0, 1, inter_frames + 2)

        for _ in tqdm(range(len(remains)), 'Generating in-between frames'):
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)
            end_i = start_i + 1

            x0 = results[start_i]
            x1 = results[end_i]

            if self.gpu and torch.cuda.is_available():
                if self.half:
                    x0 = x0.half()
                    x1 = x1.half()
                x0 = x0.cuda()
                x1 = x1.cuda()

            dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

            with torch.no_grad():
                prediction = self.model(x0, x1, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
            del remains[step]
        
        y1, x1, y2, x2 = crop_region_1
        frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]
        
        return frames
        
    def _load_model(self, model_path):
        self.model = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()

        if not self.half:
            self.model.float()

        if self.gpu and torch.cuda.is_available():
            if self.half:
                self.model = self.model.half()
            else:
                self.model.float()
            self.model = self.model.cuda()
            
    def save_animation(self, outpath, frames, fps=9):
        w, h = frames[0].shape[1::-1]
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(outpath, fourcc, fps, (w, h))
        
        if Context.revers_mode:
            for frame in frames[::-1]:
                writer.write(frame)
        else:
            for frame in frames:
                writer.write(frame)

        # for frame in frames[1:][::-1]:
        #     writer.write(frame)

        writer.release()

class AnimationMaker:
    _n = 0
    
    def __init__(self,):
        self.frames = FramesContainer()
        
        # self.frames.add_frame(Frame('frame_1', 3))
        
        with dpg.collapsing_header(label="Camera", default_open=True):
            with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                    borders_innerV=True, borders_outerV=True, row_background=True,
                    callback=None):

                dpg.add_table_column(label="up", no_resize=True, width_fixed=True)
                dpg.add_table_column(label="down", no_resize=True, width_fixed=True)
                dpg.add_table_column(label="name")
                dpg.add_table_column(label="inter")

                for i in range(10):
                    with dpg.table_row():
                        dpg.add_button(label="Button", callback=None, arrow=True, direction=dpg.mvDir_Up)  # default direction is mvDir_Up
                        dpg.add_button(label="Button", callback=None, arrow=True, direction=dpg.mvDir_Down)
                        dpg.add_text(f"Cell {i}, 1")
                        dpg.add_input_int(default_value=1)
                        
        # dpg.add_button('Add frame', callback=self.add_frame)
        
    def add_frame(self):
        pass
    
    def move_up(self, sender):
        pass
    
    def move_down(self, sender):
        pass
    
    def delete_frame(self, sender):
        pass
    
    def create_animation(self):
        pass


def reverse_callback(sender):
    val = dpg.get_value(sender)
    logger.debug(f'reverse mode changed to {val}')
    Context.revers_mode = val

class AnimationPanel:
    
    def __init__(self) -> None:
        with dpg.collapsing_header(label='Create animation'):
            # dpg.add_input_text(label='folder', tag='interp_text_input_tag', default_value='E:/GitHub/stable_points/output/run_36/')
            dpg.add_input_int(label='frames', tag='inter_frames_input', default_value=3, min_value=1, max_value=10)
            dpg.add_button(label='run_interpolation', callback=make_interpolation)
            dpg.add_input_text(label='savefile', tag='savename_input', default_value='test.mp4')
            dpg.add_input_int(label='fps', tag='fps_input', default_value=9, min_value=1, max_value=60)
            dpg.add_checkbox(label='Reverse animation', tag='reverse_checkbox', 
                             default_value=Context.revers_mode, callback=reverse_callback)
            dpg.add_button(label='Save animation', callback=save_animation)


class FramesPanel:
    
    def __init__(self) -> None:
        pass


model: FrameInterpolationModel = None
frames = []

def make_interpolation(sender):
    global model
    if model is None:
        model = FrameInterpolationModel('J:/Weights/FILM-pytorch/film_net_fp16.pt')
    
    if __name__ == "__main__":
        folder = dpg.get_value('interp_text_input_tag')
    else:
        folder = os.path.join(Context.log_folder, "render")
    inter_frames = dpg.get_value('inter_frames_input')
    
    imgs = [img for img in os.listdir(folder) if img.startswith('render')]
    imgs = sorted(imgs)
    
    frames.clear()
    for i in range(len(imgs) - 1):
        outputs = model.infer(os.path.join(folder, imgs[i]),
                              os.path.join(folder, imgs[i+1]), inter_frames)
        
        frames.extend(outputs)


def save_animation(sender):
    outfile = dpg.get_value('savename_input')
    fps = dpg.get_value('fps_input')
    print(f'Start saving animation as {outfile}')
    
    model.save_animation(outpath=outfile, frames=frames, fps=fps)

    print('Saving end')


def test():
    dpg.create_context()
    dpg.create_viewport()
    dpg.setup_dearpygui()

    def sort_callback(sender, sort_specs):

        # sort_specs scenarios:
        #   1. no sorting -> sort_specs == None
        #   2. single sorting -> sort_specs == [[column_id, direction]]
        #   3. multi sorting -> sort_specs == [[column_id, direction], [column_id, direction], ...]
        #
        # notes:
        #   1. direction is ascending if == 1
        #   2. direction is ascending if == -1

        # no sorting case
        if sort_specs is None: return

        rows = dpg.get_item_children(sender, 1)

        # create a list that can be sorted based on first cell
        # value, keeping track of row and value used to sort
        sortable_list = []
        for row in rows:
            first_cell = dpg.get_item_children(row, 1)[0]
            sortable_list.append([row, dpg.get_value(first_cell)])

        def _sorter(e):
            return e[1]

        sortable_list.sort(key=_sorter, reverse=sort_specs[0][1] < 0)

        # create list of just sorted row ids
        new_order = []
        for pair in sortable_list:
            new_order.append(pair[0])

        dpg.reorder_items(sender, 1, new_order)

    with dpg.window(label="Tutorial", width=500):

        with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                    borders_innerV=True, borders_outerV=True, row_background=True,
                    sortable=True, callback=sort_callback):

            dpg.add_table_column(label="One")
            dpg.add_table_column(label="Two", no_sort=True)

            for i in range(25):
                with dpg.table_row():
                    dpg.add_input_int(label=" ", step=0, default_value=i)
                    dpg.add_text(f"Cell {i}, 1")

    with dpg.window(label='test table', width=640):
        AnimationMaker()
        
    with dpg.window(label='Interpolation Test', width=480):
        dpg.add_input_text(label='folder', tag='interp_text_input_tag', default_value='E:/GitHub/stable_points/output/run_36/')
        dpg.add_input_int(label='frames', tag='inter_frames_input', default_value=3, min_value=1, max_value=10)
        dpg.add_button(label='run_interpolation', callback=make_interpolation)
        dpg.add_input_text(label='savefile', tag='savename_input', default_value='test.mp4')
        dpg.add_input_int(label='fps', tag='fps_input', default_value=9, min_value=1, max_value=60)
        dpg.add_button(label='Save animation', callback=save_animation)
    
    # main loop
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

    
if __name__ == "__main__":
    test()