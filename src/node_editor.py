#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import copy
import json
import asyncio
import argparse
from collections import OrderedDict
import os

import cv2
import dearpygui.dearpygui as dpg

from pathlib import Path
from loguru import logger

try:
    from .node_editor_window import DpgNodeEditor
except ImportError:
    from node_editor_window import DpgNodeEditor
    
    
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setting",
        type=str,
        # get abs
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         '../config/settings.json')),
    )
    parser.add_argument("--unuse_async_draw", action="store_true")
    parser.add_argument("--use_debug_print", action="store_true")

    args = parser.parse_args()

    return args


def async_main(node_editor):
    # 各ノードの処理結果保持用Dict
    node_image_dict = {}
    node_result_dict = {}

    # メインループ
    while not node_editor.get_terminate_flag():
        update_node_info(node_editor, node_image_dict, node_result_dict)


def update_node_info(
    node_editor,
    node_image_dict,
    node_result_dict,
    mode_async=True,
):
    # ノードリスト取得
    node_list = node_editor.get_node_list()

    # ノード接続情報取得
    sorted_node_connection_dict = node_editor.get_sorted_node_connection()

    # 各ノードの情報をアップデート
    for node_id_name in node_list:
        if node_id_name not in node_image_dict:
            node_image_dict[node_id_name] = None

        node_id, node_name = node_id_name.split(':')
        connection_list = sorted_node_connection_dict.get(node_id_name, [])

        # ノード名からインスタンスを取得
        node_instance = node_editor.get_node_instance(node_name)

        # 指定ノードの情報を更新
        if mode_async:
            try:
                image, result = node_instance.update(
                    node_id,
                    connection_list,
                    node_image_dict,
                    node_result_dict,
                )
            except Exception as e:
                print(e)
                sys.exit()
        else:
            image, result = node_instance.update(
                node_id,
                connection_list,
                node_image_dict,
                node_result_dict,
            )
        node_image_dict[node_id_name] = copy.deepcopy(image)
        node_result_dict[node_id_name] = copy.deepcopy(result)


def main():

    args = get_args()
    setting = args.setting
    unuse_async_draw = args.unuse_async_draw
    use_debug_print = args.use_debug_print

    # Load config and setup vars
    print('**** Load Config ********')
    setting_dict = None
    with open(setting) as fp:
        setting_dict = json.load(fp)

    logger.info(setting_dict)

    # DearPyGui準備(コンテキスト生成、セットアップ、ビューポート生成)
    editor_width = setting_dict['editor_width']
    editor_height = setting_dict['editor_height']

    print('**** DearPyGui Setup ********')
    dpg.create_context()
    dpg.setup_dearpygui()
    dpg.create_viewport(
        title="Image Processing Node Editor",
        width=editor_width,
        height=editor_height,
    )

    current_path = os.path.dirname(os.path.abspath(__file__))
    with dpg.font_registry():
        
        font_path = Path(current_path) / '../fonts/YasashisaAntiqueFont/07YasashisaAntique.otf'
        logger.info(f'{font_path=}')
        with dpg.font(
                str(font_path),
                16,
        ) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)
    dpg.bind_font(default_font)

    # ノードエディター生成
    print('**** Create NodeEditor ********')
    menu_dict = OrderedDict({
        'InputNode': 'input_node',
        'ProcessNode': 'process_node',
        'DeepLearningNode': 'deep_learning_node',
        'AnalysisNode': 'analysis_node',
        'MaskProcessingNode': 'mask_node',
        'DrawNode': 'draw_node',
        'OtherNode': 'other_node',
        'StableDiffusion': 'sd_nodes'
        # 'PreviewReleaseNode': 'preview_release_node'
    })
    # print
    node_editor = DpgNodeEditor(
        width=editor_width - 15,
        height=editor_height - 40,
        setting_dict=setting_dict,
        menu_dict=menu_dict,
        use_debug_print=use_debug_print,
        node_dir=current_path + '/nodes',
    )

    # ビューポート表示
    dpg.show_viewport()

    # メインループ
    print('**** Start Main Event Loop ********')
    if not unuse_async_draw:
        event_loop = asyncio.get_event_loop()
        event_loop.run_in_executor(None, async_main, node_editor)
        dpg.start_dearpygui()
    else:
        # 各ノードの処理結果保持用Dict
        node_image_dict = {}
        node_result_dict = {}
        while dpg.is_dearpygui_running():
            update_node_info(
                node_editor,
                node_image_dict,
                node_result_dict,
                mode_async=False,
            )
            dpg.render_dearpygui_frame()

    print('**** Terminate process ********')
    print('**** Close All Node ********')
    node_list = node_editor.get_node_list()
    for node_id_name in node_list:
        node_id, node_name = node_id_name.split(':')
        node_instance = node_editor.get_node_instance(node_name)
        node_instance.close(node_id)

    # イベントループの停止
    print('**** Stop Event Loop ********')
    node_editor.set_terminate_flag()
    event_loop.stop()
    # DearPyGuiコンテキスト破棄
    print('**** Destroy DearPyGui Context ********')
    dpg.destroy_context()


if __name__ == '__main__':
    main()
