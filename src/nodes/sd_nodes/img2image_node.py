#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from utils import dpg_get_value, dpg_set_value

from node_abc import DpgNodeABC
from utils import convert_cv_to_dpg

from sd_scripts.sd_model import SDModel
# from node.deep_learning_node.monocular_depth_estimation.FSRE_Depth.fsre_depth import FSRE_Depth
# from node.deep_learning_node.monocular_depth_estimation.HR_Depth.hr_depth import HR_Depth
import random
random.random()

from enum import Enum, auto
from functools import partial
from sd_scripts.inpaint_model import Inpainter


class PredictStatus(Enum):
    WAIT = auto()
    READY = auto()
    PROCESS = auto()


class Node(DpgNodeABC):
    _ver = '0.0.1'

    node_label = 'SD image2image'
    node_tag = 'sd_img2img'

    _opencv_setting_dict = None

    _min_height = 512
    _max_height = 2048
    _default_height = 768
    
    _min_width = 512
    _max_width = 2048
    _default_width = 768
    
    _min_steps = 5
    _max_steps = 150
    _default_steps = 20
    
    _min_scale = 1.0
    _max_scale = 28.0
    _default_scale = 7.0
    
    # モデル設定
    _model_class = {
        'Stable diffusion': SDModel,
        # 'FSRE-Depth(320x192)': FSRE_Depth,
        # 'FSRE-Depth(640x384)': FSRE_Depth,
        # 'Lite-HR-Depth(1280x384)': HR_Depth,
        # 'HR-Depth(1280x384)': HR_Depth,
    }
    _sampler_class = {
        'DDIM': 1,
        'PLMS': 2,
    }
    _model_base_path = os.path.dirname(os.path.abspath(__file__)) + '/monocular_depth_estimation/'
    _model_path_setting = {
        'FSRE-Depth(320x192)':
        _model_base_path +
        'FSRE_Depth/fsre_depth_192x320/fsre_depth_full_192x320.onnx',
        'FSRE-Depth(640x384)':
        _model_base_path +
        'FSRE_Depth/fsre_depth_384x640/fsre_depth_full_384x640.onnx',
        'Lite-HR-Depth(1280x384)':
        _model_base_path +
        'HR_Depth/saved_model_lite_hr_depth_384x1280/lite_hr_depth_k_t_encoder_depth_384x1280.onnx',
        'HR-Depth(1280x384)':
        _model_base_path +
        'HR_Depth/saved_model_hr_depth_384x1280/hr_depth_k_m_depth_encoder_depth_384x1280.onnx',
    }

    _model_instance = {}
    _sd_model = None
    _status = PredictStatus.WAIT

    def __init__(self):
        pass

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        """
        Inputs:
        - sampler (drop menu)
        - text (str)
        - width (int slider)
        - height (int slider)
        - steps (int slider)
        - scale (float slider)
        """
        self.node_id = node_id
        tag_node_name = str(node_id) + ':' + self.node_tag
        # tag_node_input01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01'
        # tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01Value'
        tag_node_input01_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input01'
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'
        tag_node_input02_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        tag_node_input03_name = tag_node_name + ':' + self.TYPE_INT + ':Input03'
        tag_node_input03_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        tag_node_input04_name = tag_node_name + ':' + self.TYPE_INT + ':Input04'
        tag_node_input04_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        tag_node_input05_name = tag_node_name + ':' + self.TYPE_INT + ':Input05'
        tag_node_input05_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input05Value'
        tag_node_input06_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input06'
        tag_node_input06_value_name = tag_node_name + ':' + self.TYPE_FLOAT + ':Input06Value'
        tag_node_output01_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01'
        tag_node_output01_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        tag_node_output02_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02'
        tag_node_output02_value_name = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        tag_provider_select_name = tag_node_name + ':' + self.TYPE_TEXT + ':Provider'
        tag_provider_select_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':ProviderValue'
        
        tag_run_button_name = tag_node_name + ':Button'

        # OpenCV向け設定
        self._opencv_setting_dict = opencv_setting_dict
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        use_gpu = self._opencv_setting_dict['use_gpu']

        # 初期化用黒画像
        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = convert_cv_to_dpg(
            black_image,
            small_window_w,
            small_window_h,
        )

        # テクスチャ登録
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_window_w,
                small_window_h,
                black_texture,
                tag=tag_node_output01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )

        # ノード
        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            # 画像
            with dpg.node_attribute(
                    tag=tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(tag_node_output01_value_name)
            # Sampler
            with dpg.node_attribute(
                    tag=tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    list(self._model_class.keys()),
                    default_value=list(self._model_class.keys())[0],
                    width=small_window_w,
                    tag=tag_node_input01_value_name,
                )
            # Prompt
            with dpg.node_attribute(
                    tag=tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_input_text(
                    default_value='A painting of nerdy rodent',
                    width=small_window_w,
                    tag=tag_node_input02_value_name,
                )
            # Width
            with dpg.node_attribute(
                    tag=tag_node_input03_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_slider_int(
                    min_value=self._min_width,
                    max_value=self._max_width,
                    default_value=self._default_width,
                    width=small_window_w,
                    tag=tag_node_input03_value_name,
                )
            # Height
            with dpg.node_attribute(
                    tag=tag_node_input04_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_slider_int(
                    min_value=self._min_height,
                    max_value=self._max_height,
                    default_value=self._default_height,
                    width=small_window_w,
                    tag=tag_node_input04_value_name,
                )
            # Steps
            with dpg.node_attribute(
                    tag=tag_node_input05_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_slider_int(
                    min_value=self._min_steps,
                    max_value=self._max_steps,
                    default_value=self._default_steps,
                    width=small_window_w,
                    tag=tag_node_input05_value_name,
                )
            # Steps
            with dpg.node_attribute(
                    tag=tag_node_input06_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_slider_float(
                    min_value=self._min_scale,
                    max_value=self._max_scale,
                    default_value=self._default_scale,
                    width=small_window_w,
                    tag=tag_node_input06_value_name,
                )
            # Launch button
            with dpg.node_attribute(
                    tag=tag_run_button_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                self.run_button = dpg.add_button(label='run', tag='run_txt2img', 
                                                 callback=self.generate_image)
            # 処理時間
            if use_pref_counter:
                with dpg.node_attribute(
                        tag=tag_node_output02_name,
                        attribute_type=dpg.mvNode_Attr_Output,
                ):
                    dpg.add_text(
                        tag=tag_node_output02_value_name,
                        default_value='elapsed time(ms)',
                    )

        # self._predictor = Txt2Img()
        self.result = {}
        return tag_node_name
    
    def generate_image(self):
        self._status = PredictStatus.PROCESS
    
    def _generate_image(self):
        node_id = self.node_id
        print('**** Generate ***')
        tag_node_name = str(node_id) + ':' + self.node_tag
        
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        input_value03_tag = tag_node_name + ':' + self.TYPE_INT + ':Input03Value'
        input_value04_tag = tag_node_name + ':' + self.TYPE_INT + ':Input04Value'
        input_value05_tag = tag_node_name + ':' + self.TYPE_INT + ':Input05Value'
        input_value06_tag = tag_node_name + ':' + self.TYPE_FLOAT + ':Input06Value'
        
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'
        
        print(f'{input_value02_tag=}')
        print(f'{output_value01_tag=}')
        print(f'{output_value02_tag=}')
        
        # dpg.configure_item(self.run_button, enabled=False)
        
        sampler = 'ddim'
        prompt = dpg_get_value(input_value02_tag)
        width = dpg_get_value(input_value03_tag)
        height = dpg_get_value(input_value04_tag)
        steps = dpg_get_value(input_value05_tag)
        scale = dpg_get_value(input_value06_tag)
        
        print('**** Start generation ****')
        image = self._model_instance.inpaint(
            Image.fromarray(image_resized),
            Image.fromarray(inpaint_mask),
            prompt, seed, scale, steps,
            1, w=width, h=height)
        
        print('**** Finish generation ****')
        
        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        
        texture = convert_cv_to_dpg(
            np.array(image),
            small_window_w,
            small_window_h,
        )
        dpg_set_value(output_value01_tag, texture)
        print('**** Output value was updated ****')
        
        return np.array(image)

    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
    ):
        # print('**** Update txt2img model ***')
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'
        output_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Output01Value'
        output_value02_tag = tag_node_name + ':' + self.TYPE_TIME_MS + ':Output02Value'

        tag_provider_select_value_name = tag_node_name + ':' + self.TYPE_IMAGE + ':ProviderValue'

        if self._sd_model is None:
            config = 'G:/GitHub/stable-diffusion/configs/stable-diffusion/v1-inpainting-inference.yaml'
            ckpt = 'G:/Weights/stable-diffusion/sd-v1-5-inpainting.ckpt'
            self._model_instance = Inpainter(ckpt, config, device='cuda')

        if self._status == PredictStatus.PROCESS:
            image = self._generate_image()

            self._status = PredictStatus.READY
            self.result['depth_map'] = image
            # return image, self.result
        # small_window_w = self._opencv_setting_dict['process_width']
        # small_window_h = self._opencv_setting_dict['process_height']
        # use_pref_counter = self._opencv_setting_dict['use_pref_counter']
        # use_gpu = self._opencv_setting_dict['use_gpu']

        # # 接続情報確認
        # connection_info_src = ''
        # for connection_info in connection_list:
        #     connection_type = connection_info[0].split(':')[2]
        #     if connection_type == self.TYPE_INT:
        #         # 接続タグ取得
        #         source_tag = connection_info[0] + 'Value'
        #         destination_tag = connection_info[1] + 'Value'
        #         # 値更新
        #         input_value = int(dpg_get_value(source_tag))
        #         input_value = max([self._min_val, input_value])
        #         input_value = min([self._max_val, input_value])
        #         dpg_set_value(destination_tag, input_value)
        #     if connection_type == self.TYPE_IMAGE:
        #         # 画像取得元のノード名(ID付き)を取得
        #         connection_info_src = connection_info[0]
        #         connection_info_src = connection_info_src.split(':')[:2]
        #         connection_info_src = ':'.join(connection_info_src)

        # # 画像取得
        # frame = node_image_dict.get(connection_info_src, None)

        # # CPU/GPU選択状態取得
        # provider = 'CPU'
        # if use_gpu:
        #     provider = dpg_get_value(tag_provider_select_value_name)
        
        # model_name = dpg_get_value(input_value02_tag)
        # model_path = self._model_path_setting[model_name]
        # model_class = self._model_class[model_name]

        # model_name_with_provider = model_name + '_' + provider

        # # モデル取得
        # if frame is not None:
        #     if model_name_with_provider not in self._model_instance:
        #         if provider == 'CPU':
        #             providers = ['CPUExecutionProvider']
        #             self._model_instance[
        #                 model_name_with_provider] = model_class(
        #                     model_path,
        #                     providers=providers,
        #                 )
        #         else:
        #             self._model_instance[
        #                 model_name_with_provider] = model_class(model_path)

        # # 計測開始
        # if frame is not None and use_pref_counter:
        #     start_time = time.perf_counter()

        # result = {}
        # if frame is not None:
        #     depth_map = self._model_instance[model_name_with_provider](frame)
        #     frame = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        #     result['depth_map'] = depth_map

        # # 計測終了
        # if frame is not None and use_pref_counter:
        #     elapsed_time = time.perf_counter() - start_time
        #     elapsed_time = int(elapsed_time * 1000)
        #     dpg_set_value(output_value02_tag,
        #                   str(elapsed_time).zfill(4) + 'ms')

        # # 描画
        # if frame is not None:
        #     texture = convert_cv_to_dpg(
        #         frame,
        #         small_window_w,
        #         small_window_h,
        #     )
        #     dpg_set_value(output_value01_tag, texture)

        return self.result.get('depth_map'), self.result

    def close(self, node_id):
        pass

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'

        # 選択モデル
        model_name = dpg_get_value(input_value02_tag)

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[input_value02_tag] = model_name

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value02_tag = tag_node_name + ':' + self.TYPE_TEXT + ':Input02Value'

        model_name = setting_dict[input_value02_tag]

        dpg_set_value(input_value02_tag, model_name)
