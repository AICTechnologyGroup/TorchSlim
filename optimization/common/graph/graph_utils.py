import logging
import queue
import re
from collections import defaultdict


import torch
from torch.utils.tensorboard._pytorch_graph import NodePy, NodePyIO, NodePyOP, GraphPy
from .graph import *


_logger = logging.getLogger(__name__)



class TorchProtoGraph(TorchGraph):

    def __init__(self, model, dummy_input, verbose = False):
        super().__init__(model, dummy_input)

        from tensorboard.compat.proto.config_pb2 import RunMetadata
        from tensorboard.compat.proto.graph_pb2 import GraphDef
        from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
        from tensorboard.compat.proto.versions_pb2 import VersionDef

        list_of_nodes = self.parse(self.trace.graph, self.trace, dummy_input)
        if verbose:
            print(self.trace.graph)
        self.stepstats = RunMetadata(step_stats = StepStats(dev_stats = [DeviceStepStats(device = "/device:CPU:0")]))
        self.graph_def = GraphDef(node = list_of_nodes, version = VersionDef(producer=22))

    def parse(self, graph, trace, args = None, omit_useless_nodes = True):
        nodes_py = GraphPy()
        for node in graph.inputs():
            if omit_useless_nodes:
                if not node.use():
                    continue
        if node.type().kind() != CLASSTYPE_KIND:
            nodes_py.append(NodePyIO(node, 'input'))
        
        attr_to_scope = dict()

        def node_to_name(d):
            return str(d).split(":")[0].strip()
        
        for node in graph.nodes():
            if node.kind() == GETATTR_KIND:
                attr_name = node.s('name')
                node_name = node_to_name(node)
                parent = node.input().node()

                if parent.kind() ==  GETATTR_KIND:
                    parent_scope = attr_to_scope[node_to_name(parent)]
                    attr_scope = parent_scope.split("/")[-1]
                    attr_to_scope[node_name] = '{}/{}.{}'.format(parent_scope, attr_scope, attr_name)
                else:
                    attr_to_scope[node_name] = "__module.{}".format(attr_name)
                
                if node.output().type().kind() != CLASSTYPE_KIND:
                    node_py = NodePyOP(node)
                    node_py.scopeName = attr_to_scope[node_name]
                    nodes_py.append(node_py)
            else:
                nodes_py.append(NodePyOP(node))
        
        for i, node in enumerate(graph.outputs()):
            node_py = NodePyIO(node, 'output')
            node_py.debugName = "output.{}".format(i + 1)
            node_py.inputs = [node.debugName()]
            nodes_py.append(node_py)
        
        alias_to_name = dict()
        base_name = parse_traced_name(trace._name)
        for name, module in trace.named_modules(prefix='__module'):
            mod_name = parse_traced_name(module._name)
            attr_name = name.split('.')[-1]
            alias_to_name[name] = '{}[{}]'.format(mod_name, attr_name)

        for node in nodes_py.nodes_op:
            module_aliases = node.scopeName.split('/')[-1].split('.')
            module_name = ''
            for i, alias in enumerate(module_aliases):
                if i == 0:
                    module_name = alias
                    node.scopeName = base_name
                else:
                    module_name += '.' + alias
                    node.scopeName += '/' + \
                        (alias_to_name[module_name]
                         if module_name in alias_to_name else alias)

        nodes_py.populate_namespace_from_OP_to_IO()
        return nodes_py.to_proto()


class NodePyGroup(NodePy):

    def __init__(self, name, unique_name, node_type, op_type, node_cpps, inputs=None, outputs=None, key_node=None):

        super(NodePyGroup, self).__init__(name, [])
        self.node_cpps = node_cpps
        self.name = name
        self.unique_name = unique_name
        self.op_type = op_type
        self.type = node_type
        self.nodes = []
        self.auxiliary = None
        self.add_nodes(node_cpps)
        self.inputs = inputs
        self.outputs = outputs
        self.key_node = key_node

    
    def add_nodes(self, node_cpps):

        for node_cpp in node_cpps:
            nodepy = NodePyOP(node_cpp)
            nodepy.name = node_cpp.scopeName() + "_" + node_cpp.kind()
            self.nodes.append(nodepy)
        
    
    def sub_node_names(self):
        return [x.name for x in self.nodes]

    def __repr__(self):
        return 'name: {}, type: {}, op_type: {}, sub_nodes: {}, inputs: {}, outputs: {}, aux: {}'.format(self.name, self.type, self.op_type, self.sub_node_names(), self.inputs, self.outputs, self.auxiliary)
    

class TorchModuleGraph(TorchGraph):

    def __init__(self, model=None, dummy_input = None, traced_model=None):
        super().__init__(model, dummy_input, traced_model)
        self.global_count  = 0
        self.reused_module = set()
        self.name_to_node, self.input_to_node, self.output_to_node = self._build_graph()
        self._extract_auxiliary_info()

    
    def _expand_key_func_node(self, node, nodes, output_to_node, module_type):

        node_name = '.'.join([self._get_module_name(node.scopeName()), node.kind(), str(self.global_count)])
        unique_name = node_name
        _logger.debug("expand non-prim node, node name: %s", node_name)
        self.global_count += 1
        op_type = node.kind()
        node_group = [node]
        inputs = []
        outputs = []
        node_queue = queue.Queue()
        node_queue.put(node)
        while not node_queue.empty():
            curr_node = node_queue.get()
            for _input in curr_node.inputs():
                if _input.node().kind() == CONSTANT_KIND:
                    continue
                input_name = _input.debugName()
                if input_name in output_to_node:
                    for predecessor_node in output_to_node[input_name]:
                        if predecessor_node in nodes:
                            if not self._is_key_func(predecessor_node):
                                if predecessor_node not in node_group:
                                    node_group.append(predecessor_node)
                                    node_queue.put(predecessor_node)
                            else:
                                inputs.append(input_name)
                        else:
                            inputs.append(input_name)
                else:
                    inputs.append(input_name)
        for output in node.outputs():
            if output.node().kind() == CONSTANT_KIND:
                continue
            outputs.append(output.debugName())
        nodepy = NodePyGroup(node_name, unique_name, module_type, op_type,
                             node_group, inputs=inputs, outputs=outputs, key_node=node)
        return nodepy

    def _expand_module_node(self, node, node_name, unique_name, op_type, nodes, input_to_node, output_to_node, module_type):

        _logger.debug("expand module node, node name: %s", node_name)
        self.global_count += 1
        if not op_type:
            op_type = node.kind()
        node_group = [node]
        inputs = []
        outputs = []
        node_queue = queue.Queue()
        node_queue.put(node)
        visited = {node}
        while not node_queue.empty():
            curr_node = node_queue.get()
            for _input in curr_node.inputs():
                if _input.node().kind() == CONSTANT_KIND:
                    continue
                input_name = _input.debugName()
                if input_name in output_to_node:
                    for predecessor_node in output_to_node[input_name]:
                        if predecessor_node in nodes:
                            if predecessor_node not in visited:
                                node_group.append(predecessor_node)
                                node_queue.put(predecessor_node)
                                visited.add(predecessor_node)
                        else:
                            inputs.append(input_name)
                else:
                    inputs.append(input_name)
            for _output in curr_node.outputs():
                if _output.node().kind() == CONSTANT_KIND:
                    continue
                output_name = _output.debugName()
                if output_name in input_to_node:
                    for successor_node in input_to_node[output_name]:
                        if successor_node in nodes:
                            if successor_node not in visited:
                                node_group.append(successor_node)
                                node_queue.put(successor_node)
                                visited.add(successor_node)
                        else:
                            outputs.append(output_name)
                else:
                    outputs.append(output_name)
        unique_outputs = list(set(outputs))
        # remove the dumplicated output names
        unique_outputs.sort(key=outputs.index)

        nodepy = NodePyGroup(node_name, unique_name, module_type, op_type,
                             node_group, inputs=list(inputs), outputs=unique_outputs)
        return nodepy


    def _extract_cat_info(self, node_group, cpp_node):

        assert cpp_node.kind() == CAT_KIND
        cat_info = {}
        t_ouput = cpp_node.output()
        out_shape = t_ouput.type().sizes()
        cat_info['out_shape'] = out_shape
        inputs = cpp_node.inputs()
        cat_dim = list(inputs)[1].toIValue()
        cat_info['cat_dim'] = cat_dim
        input_order = []
        list_construct_cpp = list(cpp_node.inputs())[0].node()
        input_tensors = list(list_construct_cpp.inputs())
        for _tensor in input_tensors:
            debug_name = _tensor.debugName()
            if debug_name in self.output_to_node:
                input_order.append(self.output_to_node[debug_name].unique_name)
            else:
                # the input tensor may be the input tensor of the whole model
                input_order.append(None)
        cat_info['in_order'] = input_order
        input_shapes = [t.type().sizes() for t in input_tensors]
        cat_info['in_shape'] = input_shapes
        return cat_info

    def _extract_linear_shape_info(self, node_group):
        for cpp_node in node_group.node_cpps():
            t_input = list(cpp_node.inputs())[1]
            t_output = cpp_node.output()
            assert isinstance(t_input.type(), torch._C.TensorType)
            assert isinstance(t_output.type(), torch._C.TensorType)
            in_shape = t_input.type().sizes()
            out_shape = t_output.type().sizes()
            return {'in_shape': in_shape, 'out_shape': out_shape}
        return None

    def _extract_shape_info(self, node):
        t_input = None
        for _input in node.inputs():
            t_input = _input
            break
        t_output = node.output()
        assert isinstance(t_input.type(), torch._C.TensorType)
        assert isinstance(t_output.type(), torch._C.TensorType)
        in_shape = t_input.type().sizes()
        out_shape = t_output.type().sizes() 
        return {
            'input' : in_shape,
            'out_shape' : out_shape
        }

    def _extract_leaf_modules(self):

        def is_parent(name1, name2):

            parts1, parts2 = name1.split("."), name2.split(".")
            if len(parts1) >= len(parts2):
                return False
            for i, _ in enumerate(parts1):
                if parts2[i] != parts1[i]:
                    return False
            return True
        
        module_names = sorted([x[0] for x in self.trace.named_modules() if x[0]])
        leaf_nodes = []
        for i, name in enumerate(module_names):
            if i + 1 >= len(module_names) or not is_parent(name,module_names[i + 1]):
                leaf_nodes.append(name)
        return leaf_nodes

    def _get_module_names(self, scope_name):

        if torch.__version__ >= '1.4.0':
            return scope_name.split('/')[-1].replace("__module.", "")
        else:
            return ".".join(re.findall(r'\[(.*?)\]', scope_name))
    
    def _build_index(self, nodes_op):
        name_to_node = dict()
        input_to_node = defaultdict(list)
        output_to_node = dict()
        for node in nodes_op:
            name_to_node[node.unique_name] = node
            for _input in node.inputs:
                if node not in input_to_node[_input]:
                    input_to_node[_input].append(node)
            for output in node.outputs():
                if output in output_to_node:
                    assert output_to_node[output] == node, "One output cannot be generated by multiple nodes %s" % output
                output_to_node[output] = node
        return name_to_node, input_to_node, output_to_node

    def _is_key_func(self, node_cpp):
        if node_cpp.kind().startswith("aten::"):
            return True
        if node_cpp.kind() in [LIST_UNPACK_KIND, TUPLE_UNPACK_KIND]:
            return True
        return False

    def unpack_manually(self):

        if hasattr(self, "unpacked"):
            return
        for node in self.nodes_py.nodes_op:
            if node.op_type in [TUPLE_UNPACK_KIND, LIST_UNPACK_KIND]:
                unpack_cpp = node.key_node
                last_cpp = list(unpack_cpp.inputs())[0].node()
                if last_cpp.kind() in [TUPLE_CONSTRUCT_KIND, LIST_CONSTRUCT_KIND]:
                    _logger.debug('List/Tuple Construct Node(cpp) %s', str(last_cpp))
                    _logger.debug('List/Tuple Unpack Node(cpp) %s', str(unpack_cpp))
                    assert len(list(unpack_cpp.outputs())) == len(list(last_cpp.inputs()))
                    assert len(node.inputs) == len(list(last_cpp.inputs())), '%s Input number: %d if inconsistent with the output number %d' % (unpack_cpp, len(node.inputs), len(list(last_cpp.inputs())))

                    for _debug_input, _debug_output in zip(node.inputs, node.outputs):
                        if _debug_input in self.input_to_node and _debug_output in self.input_to_node:
                            if node in self.input_to_node[_debug_input]:
                                self.input_to_node[_debug_input].remove(node)

                            self.input_to_node[_debug_input].extend(self.input_to_node[_debug_output])
                
                        if _debug_output in self.input_to_node:
                            for following_node in self.input_to_node[_debug_output]:
                                _tmp_index = following_node.inputs.index(_debug_output)
                                following_node.inputs[_tmp_index] = _debug_input
        self.unpacked = True
    
    def _build_graph(self):

        omit_useless_nodes = True
        graph = self.trace.graph
        _logger.debug(graph)

        input_to_node = defaultdict(list)
        output_to_node = defaultdict(list)

        for node in graph.nodes():
            if node.kind() == CONSTANT_KIND:
                continue
            for x in node.outpus():
                if x.node().kind() == CONSTANT_KIND:
                    continue
                    ouptut_to_node[x.debugName].append(node)
                    assert len(output_to_node[x.debugName]) <= 1, "One output cannot be generated by multiple nodes %s" % x.debugName()
            for x in node.inputs():
                if x.node().kind() ==  CONSTANT_KIND:
                    continue
                input_to_node[x.debugName()].append(node)
        
        module_to_nodes = defaultdict(list)
        func_to_nodes = defaultdict(list)
        nodes_py = GraphPy()

        for node in graph.inputs():
            if omit_useless_nodes:
                if not node.uses():
                    continue
            if node.type().kind() != "ClassType":
                nodes_py.append(NodePyIO(node, 'input'))

        self.leaf_modules = self._extract_leaf_modules()
        module_to_type = {
            name : parse_traced_name(module._name) for name, module in self.trace.named_modules()
        }

        for node in graph.nodes():
            if node.kind() == CONSTANT_KIND:
                continue
            module_name = self._get_module_name(node.scopeName())
            if module_name in self.leaf_modules:
                module_to_nodes[module_name].append(node)
            else:
                func_to_nodes[node.scopeName()].append(node)
        
        for module_name, node_cpps in module_to_nodes.items():
            use_count = 0
            merged = set()
            for node in node_cpps:
                if node in merged:
                    unique_name = module_name
                    if use_count > 0:
                        unique_name = module_name + ".%d" %use_count
                        self.reused_module.add(unique_name)
                        self.reused_module.add(module_name)
                    node_group = self._expand_module_node(node, module_name, unique_name, module_to_type[module_name], node_cpps, input_to_node, output_to_node, 'module')

                    nodes_py.nodes_op.append(node_group)
                    use_count += 1
                    merged.update(node_group.node_cpps)
        
        for _, nodes in func_to_nodes.items():
            key_func_nodes = list()
            for node in nodes:
                if self._is_key_func(node):
                    key_func_nodes.append(node)
            for node in key_func_nodes:
                node_group = self._expand_key_func_node(node, nodes, input_to_node, output_to_node, 'func')
                nodes_py.nodes_op.append(node_group)

        for node in graph.outputs():
            node_py = NodePyIO(node,  'output')
            nodes_py.append(node_py)
        self.nodes_py = nodes_py

        return self._build_index(self.nodes_py.nodes_py)

    
    def _extract_auxiliary_info(self):

        for node_group in self.nodes_py.nodes_op:
            if node_group.op_type in ['aten::view', 'aten::flatten', 'aten::mean', 'aten::reshape']:
                cpp_node = list(filter(lambda x: x.kind() == node_group.op_type, node_group.node_cpps))[0]
                node_group.auxiliary = self._extract_shape_info(cpp_node)
            elif node_group.op_type == "Linear":
                node_group.auxiliary = self._extract_linear_shape_info(node_group)
            elif node_group.op_type == CAT_KIND:
                cpp_node = list(filter(lambda x: x.kind() == node_group.op_type, node_group.node_cpps))[0]

                node_group.auxiliary = self._extract_cat_info(node_group, cpp_node)
    

    def find_predecessors(self, unique_name):
        predecessors = []
        for _input in self.name_to_node[unique_name].inputs:
            if not _input in self.output_to_node:
                _logger.debug("cannot find node with %s as its output", _input)
            else:
                node_py = self.output_to_node[_input]
                predecessors.append(node_py.unique_name)
        return predecessors
    
    def find_successors(self, unique_name):
        successors = []
        for output in self.name_to_node[unique_name].outputs:
            if output not in self.input_to_node:
                continue
            nodes_py = self.input_to_node[output]
            for node_py in nodes_py:
                successors.append(node_py.unique_name)
        return successors

        
