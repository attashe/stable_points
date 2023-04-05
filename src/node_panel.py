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


