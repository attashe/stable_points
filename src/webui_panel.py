import time
import dearpygui.dearpygui as dpg
from loguru import logger
import numpy as np

from context import Context
from depth_infer import DepthModel, AdaBinsDepthPredict, LeResInfer
from utils import dpg_get_value


class WebUIPanel:
    
    def __init__(self) -> None:
        self.loaded_model = None  # 
        
        # with dpg.group(label="Depth"):
        with dpg.collapsing_header(label="WebUI"):
            dpg.add_combo(
                    ['midas', 'adabins', 'leres'],
                    default_value=Context.depth_type,
                    tag=self.selector_tag,
                    callback=self.load_model_callback
                )

    def _get_models_list(self) -> list:
        pass
    
    def _set_model(self, model):
        pass
    
    def txt2img(self):
        pass
    
    def img2img(self):
        pass
    
    def inpaint(self):
        pass
