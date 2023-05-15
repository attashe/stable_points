from typing import List, Callable, Dict
from queue import Queue

import dearpygui.dearpygui as dpg

from node_abc import DpgNodeABC
from pipelines import vortex_action, save_render_action
from render_panel import update_render_view


# Window properties
small_window_w = 200

# callback runs when user attempts to connect attributes
def link_callback(sender, app_data):
    # app_data -> (link_id1, link_id2)
    dpg.add_node_link(app_data[0], app_data[1], parent=sender)

# callback runs when user attempts to disconnect attributes
def delink_callback(sender, app_data):
    # app_data -> link_id
    dpg.delete_item(app_data)


class RenderNode(DpgNodeABC):
    
    def __init__(self):
        pass
    
    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        # callback=None,
    ):
        self.node_id = node_id
        tag_node_name = str(node_id) + ':' + self.node_tag

        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text('Render', tag=tag_node_name+':input_rgb')
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text('Output render', tag=tag_node_name+':output_rgb')

    def close(self, node_id):
        pass
    
    def update(self, node_id):
        pass
    
    def get_setting_dict(self, node_id):
        pass

    def set_setting_dict(self, node_id, setting_dict):
        pass



class VortexNode(DpgNodeABC):
    
    node_label = 'Vortex Node'
    node_tag = 'vortex_node'
    
    _opencv_setting_dict = None
    
    def __init__(self) -> None:
        pass
    
    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        self.node_id = node_id
        tag_node_name = str(node_id) + ':' + self.node_tag

        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text('Input RGB Render', tag=tag_node_name+':input_rgb')
            
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text('Output RGB Image', tag=tag_node_name+':output_rgb')
                
    def run(self):
        vortex_action()
        update_render_view()
        
    def close(self, node_id):
        pass
    
    def update(self, node_id):
        pass
    
    def get_setting_dict(self, node_id):
        pass

    def set_setting_dict(self, node_id, setting_dict):
        pass

class StartNode(DpgNodeABC):
    
    node_label = 'Start'
    
    def __init__(self) -> None:
        pass
    
    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        # callback=None,
    ):
        self.node_id = node_id
        tag_node_name = str(node_id) + ':' + self.node_tag

        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text('Start Node', tag=tag_node_name+':output')
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_input_int(label='Repeats', tag=tag_node_name+':repeats',
                                  width=small_window_w,
                                  default_value=1, min_value=1, max_value=100)
            
    def close(self, node_id):
        pass
    
    def update(self, node_id):
        pass
    
    def get_setting_dict(self, node_id):
        pass

    def set_setting_dict(self, node_id, setting_dict):
        pass

            
class EndNode(DpgNodeABC):
    
    node_label = 'End'
    
    def __init__(self) -> None:
        pass
    
    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        self.node_id = node_id
        tag_node_name = str(node_id) + ':' + self.node_tag

        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text('End Node', tag=tag_node_name+':input')
            
    def close(self, node_id):
        pass
    
    def update(self, node_id):
        pass
    
    def get_setting_dict(self, node_id):
        pass

    def set_setting_dict(self, node_id, setting_dict):
        pass


class SaveNode(DpgNodeABC):
    """Save render 
    """
    node_label = 'Save'
    
    def __init__(self) -> None:
        pass
    
    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        self.node_id = node_id
        tag_node_name = str(node_id) + ':' + self.node_tag

        with dpg.node(
                tag=tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):
            with dpg.node_attribute(tag=tag_node_name+':attr_image',
                                    attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text('Input RGB Image')
            
            with dpg.node_attribute(tag=tag_node_name+':attr_output',
                                    attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text('Output RGB Image')
    
    def run(self):
        save_render_action()
        
    def close(self, node_id):
        pass
    
    def update(self, node_id):
        pass
    
    def get_setting_dict(self, node_id):
        pass

    def set_setting_dict(self, node_id, setting_dict):
        pass



# TODO change list of inputs values to dictionary
# Возможно, что этот огород не нужен и может быть реализован через EventPropagation, но в этом не уверен
class Node:
    _all_nodes = []
    
    def __init__(self, func: Callable[[Dict], None], outputs=[], inputs=[]):
        self.outputs: List[Node] = outputs
        self.inputs: List[Node] = inputs
        self.func = func
        self.res = None
        
        # For debug
        Node._all_nodes.append(self)
        
    def add_child(self, func, inputs=[]):
        node = Node(func, outputs=[], inputs=[self,]+list(inputs))
        self.outputs.append(node)
        for n in inputs:
            n.outputs.append(node)
        return node
        
    def run(self):
        self.res = self.func([node.res for node in self.inputs])


class Tree:
    
    def __init__(self, root):
        self.root = root
    
    def clear_tree(self):
        pass
    
    def calculate(self):
        stack = list()
        
        self.root.run()
        
        stack += self.root.outputs
        
        while len(stack) > 0:
            node = stack.pop()
            
            is_none = [n for n in node.inputs if n.res is None]
            if len(is_none) > 0:
                stack.append(node)
                
                for n in is_none:
                    stack.append(n)
            else:
                if node.res is None:
                    node.run()
                    for n in node.outputs:
                        stack.append(n)

class ExecutionTree:
    
    def __init__(self, start_node) -> None:
        self.tree = Tree(start_node)
        
    def add_node(self, parent, node):
        pass
    
    def link_nodes(self, source, target):
        pass
    
    def unlink_nodes(self, source, target):
        pass
    
    def delete_node(self, parent, node):
        pass

def test():
    def foo(inputs):
        print(f'Inputs: {inputs}')
        print('Func was called')
        
        return 5

    root = Node(foo)

    c1 = root.add_child(foo)
    c2 = root.add_child(foo)

    c3 = c1.add_child(foo, inputs=(c2,))
    c4 = c2.add_child(foo)

    tree = Tree(root)
    
    for n in Node._all_nodes:
        print(n.__dict__)
    
    tree.calculate()
    
    for n in Node._all_nodes:
        print(n.__dict__)

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
                # sys.exit()
        else:
            image, result = node_instance.update(
                node_id,
                connection_list,
                node_image_dict,
                node_result_dict,
            )
        node_image_dict[node_id_name] = copy.deepcopy(image)
        node_result_dict[node_id_name] = copy.deepcopy(result)



class NodeEditorPanel:
    _node_id = 0
    
    _node_mapping: Dict[str, Node] = {
        'start': StartNode(),
        'vortex': VortexNode(),
        'save': SaveNode(),
        'render': RenderNode(),
        'end': EndNode()
    }
        
    def __init__(self) -> None:
        self.tree: Tree = None
        self._node_editor_tag = 'node_editor_tag'
        
        self._setting_dict = {}
        self._last_pos = None
        
        with dpg.collapsing_header(label='Node Editor'):
            dpg.add_text('Select node')
            dpg.add_combo(items=[
                'start', 'vortex', 'save', 'end'
            ], default_value='start', tag='node_name_combo')
            dpg.add_button(label='Add node', callback=self.add_node)
            # Button for check if scenario buitd is correct and print debug info
            dpg.add_button(label='Check node scenario', callback=self.check_node_scenario)
            dpg.add_input_int(label='Repeats', default_value=1, min_value=0, max_value=100)
            # Run nodes scenario
            dpg.add_button(label='Run', callback=self.run_scenario)
    
    def add_node(self, sender):
        # https://www.reddit.com/r/DearPyGui/comments/sh410p/how_to_spawn_new_nodes_in_the_node_editor/
        self._node_id += 1

        node_name = dpg.get_value('node_name_combo')
        node = self._node_mapping[node_name]
        print(f'Node name is {node_name} type {type(node)}')

        self._callback_save_last_pos()
        
        # TODO: сформировать _settings_dict и обновлять его перед созданием новой ноды
        # self._setting_dict.update(...)

        last_pos = [0, 0]
        if self._last_pos is not None:
            last_pos = [self._last_pos[0] + 30, self._last_pos[1] + 30]
        tag_name = node.add_node(
            self._node_editor_tag,
            self._node_id,
            pos=last_pos,
            opencv_setting_dict=self._setting_dict,
        )
        
    def _callback_save_last_pos(self):
        print('Update last position callback')
        if len(dpg.get_selected_nodes(self._node_editor_tag)) > 0:
            self._last_pos = dpg.get_item_pos(
                dpg.get_selected_nodes(self._node_editor_tag)[0])
            
    def check_node_scenario(self):
        pass
    
    def run_scenario(self):
        pass
    
    def get_node_list(self):
        return self._node_list

    def get_sorted_node_connection(self):
        return self._node_connection_dict

    def get_node_instance(self, node_name):
        return self._node_instance_list.get(node_name, None)

    def set_terminate_flag(self, flag=True):
        self._terminate_flag = flag

    def get_terminate_flag(self):
        return self._terminate_flag

    def _callback_add_node(self, sender, data, user_data):
        self._node_id += 1

        # ノードインスタンス取得
        node = self._node_instance_list[user_data]

        # ノードエディターにノードを追加
        last_pos = [0, 0]
        if self._last_pos is not None:
            last_pos = [self._last_pos[0] + 30, self._last_pos[1] + 30]
        tag_name = node.add_node(
            self._node_editor_tag,
            self._node_id,
            pos=last_pos,
            opencv_setting_dict=self._setting_dict,
        )

        self._node_list.append(tag_name)

        if self._use_debug_print:
            print('**** _callback_add_node ****')
            print('    Node ID         : ' + str(self._node_id)
                  )
            print('    sender          : ' + str(sender))
            print('    data            : ' + str(data))
            print('    user_data       : ' + str(user_data))
            print('    self._node_list : ' + ', '.join(self._node_list))
            print()

    def _callback_link(self, sender, data):
        # 各接続子の型を取得
        source_type = data[0].split(':')[2]
        destination_type = data[1].split(':')[2]

        # 型が一致するもののみ処理
        if source_type == destination_type:
            # 初回ノード生成時
            if len(self._node_link_list) == 0:
                dpg.add_node_link(data[0], data[1], parent=sender)
                self._node_link_list.append([data[0], data[1]])
            # 2回目以降
            else:
                # 入力端子に複数接続しようとしていないかチェック
                duplicate_flag = False
                for node_link in self._node_link_list:
                    if data[1] == node_link[1]:
                        duplicate_flag = True
                if not duplicate_flag:
                    dpg.add_node_link(data[0], data[1], parent=sender)
                    self._node_link_list.append([data[0], data[1]])

        # ノードグラフ再生成
        self._node_connection_dict = self._sort_node_graph(
            self._node_list,
            self._node_link_list,
        )

        if self._use_debug_print:
            print('**** _callback_link ****')
            print('    sender                     : ' + str(sender))
            print('    data                       : ' + ', '.join(data))
            print('    self._node_list            :    ', self._node_list)
            print('    self._node_link_list       : ', self._node_link_list)
            print('    self._node_connection_dict : ',
                  self._node_connection_dict)
            print()

    def _callback_delink(self, sender, data):
        # リンクリストから削除
        self._node_link_list.remove([
            dpg.get_item_configuration(data)['attr_1'],
            dpg.get_item_configuration(data)['attr_2']
        ])

        # ノードグラフ再生成
        self._node_connection_dict = self._sort_node_graph(
            self._node_list,
            self._node_link_list,
        )

        # アイテム削除
        dpg.delete_item(data)

        if self._use_debug_print:
            print('**** _callback_delink ****')
            print('    sender                     : ' + str(sender))
            print('    data                       : ' + str(data))
            print('    self._node_list            :    ', self._node_list)
            print('    self._node_link_list       : ', self._node_link_list)
            print('    self._node_connection_dict : ',
                  self._node_connection_dict)
            print()

    def _callback_close_window(self, sender):
        dpg.delete_item(sender)

    def _sort_node_graph(self, node_list, node_link_list):
        node_id_dict = OrderedDict({})
        node_connection_dict = OrderedDict({})

        # ノードIDとノード接続を辞書形式で整理
        for node_link_info in node_link_list:
            source_id = int(node_link_info[0].split(':')[0])
            destination_id = int(node_link_info[1].split(':')[0])

            if destination_id not in node_id_dict:
                node_id_dict[destination_id] = [source_id]
            else:
                node_id_dict[destination_id].append(source_id)

            source = node_link_info[0]
            destination = node_link_info[1]
            split_destination = destination.split(':')

            node_name = split_destination[0] + ':' + split_destination[1]
            if node_name not in node_connection_dict:
                node_connection_dict[node_name] = [[source, destination]]
            else:
                node_connection_dict[node_name].append([source, destination])

        node_id_list = list(node_id_dict.items())
        node_connection_list = list(node_connection_dict.items())

        # 入力から出力に向かって処理順序を入れ替える
        index = 0
        while index < len(node_id_list):
            swap_flag = False
            for check_id in node_id_list[index][1]:
                for check_index in range(index + 1, len(node_id_list)):
                    if node_id_list[check_index][0] == check_id:
                        node_id_list[check_index], node_id_list[
                            index] = node_id_list[index], node_id_list[
                                check_index]
                        node_connection_list[
                            check_index], node_connection_list[
                                index] = node_connection_list[
                                    index], node_connection_list[check_index]

                        swap_flag = True
                        break
            if not swap_flag:
                index += 1

        # 接続リストに登場しないノードを追加する(入力ノード等)
        index = 0
        unfinded_id_dict = {}
        while index < len(node_id_list):
            for check_id in node_id_list[index][1]:
                check_index = 0
                find_flag = False
                while check_index < len(node_id_list):
                    if check_id == node_id_list[check_index][0]:
                        find_flag = True
                        break
                    check_index += 1
                if not find_flag:
                    for index, node_id_name in enumerate(node_list):
                        node_id, node_name = node_id_name.split(':')
                        if node_id == check_id:
                            unfinded_id_dict[check_id] = node_id_name
                            break
            index += 1

        for unfinded_value in unfinded_id_dict.values():
            node_connection_list.insert(0, (unfinded_value, []))

        return OrderedDict(node_connection_list)

    def _callback_file_export(self, sender, data):
        setting_dict = {}

        # ノードリスト、接続リスト保存
        setting_dict['node_list'] = self._node_list
        setting_dict['link_list'] = self._node_link_list

        # 各ノードの設定値保存
        for node_id_name in self._node_list:
            node_id, node_name = node_id_name.split(':')
            node = self._node_instance_list[node_name]

            setting = node.get_setting_dict(node_id)

            setting_dict[node_id_name] = {
                'id': str(node_id),
                'name': str(node_name),
                'setting': setting
            }

        # JSONファイルへ書き出し
        with open(data['file_path_name'], 'w') as fp:
            json.dump(setting_dict, fp, indent=4)

        if self._use_debug_print:
            print('**** _callback_file_export ****')
            print('    sender          : ' + str(sender))
            print('    data            : ' + str(data))
            print('    setting_dict    : ', setting_dict)
            print()

    def _callback_file_export_menu(self):
        dpg.show_item('file_export')

    def _callback_file_import_menu(self):
        if self._node_id == 0:
            dpg.show_item('file_import')
        else:
            dpg.configure_item('modal_file_import', show=True)

    def _callback_file_import(self, sender, data):
        if data['file_name'] != '.':
            # JSONファイルから読み込み
            setting_dict = None
            with open(data['file_path_name']) as fp:
                setting_dict = json.load(fp)

            # 各ノードの設定値復元
            for node_id_name in setting_dict['node_list']:
                node_id, node_name = node_id_name.split(':')
                node = self._node_instance_list[node_name]

                node_id = int(node_id)

                if node_id > self._node_id:
                    self._node_id = node_id

                # ノードインスタンス取得
                node = self._node_instance_list[node_name]

                # バージョン警告
                ver = setting_dict[node_id_name]['setting']['ver']
                if ver != node._ver:
                    warning_node_name = setting_dict[node_id_name]['name']
                    print('WARNING : ' + warning_node_name, end='')
                    print(' is different version')
                    print('                     Load Version ->' + ver)
                    print('                     Code Version ->' + node._ver)
                    print()

                # ノードエディターにノードを追加
                pos = setting_dict[node_id_name]['setting']['pos']
                node.add_node(
                    self._node_editor_tag,
                    node_id,
                    pos=pos,
                    opencv_setting_dict=self._setting_dict,
                )

                # 設定値復元
                node.set_setting_dict(
                    node_id,
                    setting_dict[node_id_name]['setting'],
                )

            # ノードリスト、接続リスト復元
            self._node_list = setting_dict['node_list']
            self._node_link_list = setting_dict['link_list']

            # ノード接続復元
            for node_link in self._node_link_list:
                dpg.add_node_link(
                    node_link[0],
                    node_link[1],
                    parent=self._node_editor_tag,
                )

            # ノードグラフ再生成
            self._node_connection_dict = self._sort_node_graph(
                self._node_list,
                self._node_link_list,
            )

        if self._use_debug_print:
            print('**** _callback_file_import ****')
            print('    sender          : ' + str(sender))
            print('    data            : ' + str(data))
            print('    setting_dict    : ', setting_dict)
            print()

    def _callback_save_last_pos(self):
        if len(dpg.get_selected_nodes(self._node_editor_tag)) > 0:
            self._last_pos = dpg.get_item_pos(
                dpg.get_selected_nodes(self._node_editor_tag)[0])

    def _callback_mv_key_del(self):
        if len(dpg.get_selected_nodes(self._node_editor_tag)) > 0:
            # 選択中のノードのアイテムIDを取得
            item_id = dpg.get_selected_nodes(self._node_editor_tag)[0]
            # ノード名を特定
            node_id_name = dpg.get_item_alias(item_id)
            node_id, node_name = node_id_name.split(':')

            if node_name != 'ExecPythonCode':
                # ノード終了処理
                node_instance = self.get_node_instance(node_name)
                node_instance.close(node_id)
                # ノードリストから削除
                self._node_list.remove(node_id_name)
                # ノードリンクリストから削除
                copy_node_link_list = copy.deepcopy(self._node_link_list)
                for link_info in copy_node_link_list:
                    source_node = link_info[0].split(':')[:2]
                    source_node = ':'.join(source_node)
                    destination_node = link_info[1].split(':')[:2]
                    destination_node = ':'.join(destination_node)

                    if source_node == node_id_name or destination_node == node_id_name:
                        self._node_link_list.remove(link_info)

                # ノードグラフ再生成
                self._node_connection_dict = self._sort_node_graph(
                    self._node_list,
                    self._node_link_list,
                )

                # アイテム削除
                dpg.delete_item(item_id)

        if self._use_debug_print:
            print('**** _callback_mv_key_del ****')
            print('    self._node_list            :    ', self._node_list)
            print('    self._node_link_list       : ', self._node_link_list)
            print('    self._node_connection_dict : ',
                  self._node_connection_dict)


class NodeEditorView:
    
    def __init__(self) -> None:
        with dpg.node_editor(tag='node_editor_tag', height=400,
                             callback=link_callback, delink_callback=delink_callback):
            with dpg.node(label="Node 1"):
                with dpg.node_attribute(label="Node A1"):
                    dpg.add_input_float(label="F1", width=150)

                with dpg.node_attribute(label="Node A2", attribute_type=dpg.mvNode_Attr_Output):
                    dpg.add_input_float(label="F2", width=150)

            with dpg.node(label="Node 2"):
                with dpg.node_attribute(label="Node A3"):
                    dpg.add_input_float(label="F3", width=200)

                with dpg.node_attribute(label="Node A4", attribute_type=dpg.mvNode_Attr_Output):
                    dpg.add_input_float(label="F4", width=200)


