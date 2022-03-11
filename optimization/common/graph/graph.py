import torch
from torch.utils.tensorboard._pytorch_graph import NodePy, NodePyIO, NodePyOP, GraphPy

__all__ = ['TorchGraph', 'NodePyGroup', 'build_module_graph', 'build_graph', 'parse_traced_name', 'CLASSTYPE_KIND', 'GETATTR_KIND', 'CAT_KIND', 'LIST_CONSTRUCT_KIND', 'LIST_UNPACK_KIND', 'TUPLE_CONSTRUCT_KIND', 'TUPLE_UNPACK_KIND', 'CONSTANT_KIND']

CLASSTYPE_KIND = 'ClassType'
GETATTR_KIND = 'prim::GetAttr'
CAT_KIND = 'aten::cat'
LIST_CONSTRUCT_KIND = 'prim::ListConstruct'
LIST_UNPACK_KIND = 'prim::ListUnpack'
TUPLE_CONSTRUCT_KIND = 'prim::TupleConstruct'
TUPLE_UNPACK_KIND = 'prim::TupleUnpack'
CONSTANT_KIND = 'prim::Constant'


def build_module_graph(model, dummy_input):
    return TorchModuleGraph(model, dummy_input)

def build_graph(model, dummy_input, verbose = False):
    g = TorchProtoGraph(model, dummy_input, verbose)
    return g.graph_def, g.stepstats

def parse_traced_name(module_name):
    prefix = 'TracedModule['
    suffix = ']'
    if module_name.startswith(prefix) and module_name.endswith(suffix):
        module_name = module_name[len(prefix):-len(suffix)]
    return module_name

class TorchGraph:

    def __init__(self, model=None, dummy_input=None, traced_model=None):

        assert torch.__version__ >= '1.3.1'
        if traced_model is not None:
            assert isinstance(traced_model, torch.jit.TopLevelTracedModule)
            self.trace = traced_model
            # it's ok if the graph is already unpacked
            torch._C._jit_pass_inline(self.trace.graph)

        elif model is not None and dummy_input is not None:
            self.bound_model = model
            self._trace(model, dummy_input)
        else:
            raise Exception(
                'Please provide model & dummy_input or the traced_model as inputs')

    def _trace(self, model, dummy_input):
        training = model.training
        model.eval()
        kw_args = {}
        if torch.__version__ >= '1.6.0':
            kw_args['strict'] = False
        self.trace = torch.jit.trace(model, dummy_input, **kw_args)
        torch._C._jit_pass_inline(self.trace.graph)
        model.train(training)


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