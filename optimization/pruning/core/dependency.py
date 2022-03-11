import csv
import logging
import re
import torch
import numpy as np

from optimization.common.graph import TorchModuleGraph
from .pruner import PrunerModuleWrapper
from .utils import get_module_by_name


_logger = logging.getLogger(__name__)

def lcm_list(L):
    lcm = 1
    for i in L:
        lcm = np.lcm(lcm, i)
    return lcm

def gcd_list(L):
    gcd = L[0]
    for i in L:
        gcd = np.gcd(gcd, i)
    return gcd

CONV_TYPE = "aten::_convolution"
ADD_TYPES = ['aten::add', "aten::add_"]
MUL_TYPES = ['atten::mul', 'atten::mul_']
CAT_TYPE = "aten::cat"
RESHAPE_OPS = [CAT_TYPE, "aten::view", "aten::reshape", "aten::flatten", "aten::mean"]

class Dependency:
    """
    Build the graph for the model.
    """
    def __init__(self, model = None, dummy_input =None, traced_model = None):

        if traced_model is None:
            assert model is not None and dummy_input is not None
        self.graph = TorchModuleGraph(model, dummy_input, traced_model)
        self.model = model
        self.dependency = dict()
        self.build_dependency()

    def build_dependency(self):
        raise NotImplementedError()
    
    def export(self, file_path):
        raise NotImplementedError()


def reshape_break_channel_dependency(op_node):
    """
    The reshape operations such as (reshape, view, flatten) may break
    the channel dependency. We need to check the input parameters of
    these reshape operations to check if this reshape node will break
    the channel dependency. However, it's complicated to analyze the the input
    parameters for each reshape function and infer if it will break the channel
    dependency. So currently, we just check if the input channel and the output
    channel is the same, if so, then we can say the original reshape function
    doesn't want to change the number of the channels, which means the channel
    dependency is not broken. In contrast, the original reshap operation wants
    to change the number of channels, so it breaks the channel dependency.

    Parameters
    ----------
    opnode: NodePyOP
        A Op node of the graph.
    Returns
    -------
    bool
        If this operation will break the channel dependency.
    """
    in_shape = op_node.auxiliary['in_shape']
    out_shape = op_node.auxiliary['out_shape']
    in_channel = in_shape[1]
    out_channel = out_shape[1]
    return in_channel != out_channel


class ChannelDependency(Dependency):

    def __init__(self, model, dummy_input, traced_model = None, prune_type="Filter"):
        """
        This model analyze the channel dependencies between the conv
        layers in a model.
        Parameters
        ----------
        model : torch.nn.Module
            The model to be analyzed.
        data : torch.Tensor
            The example input data to trace the network architecture.
        traced_model : torch._C.Graph
            if we alreay has the traced graph of the target model, we donnot
            need to trace the model again.
        prune_type: str
            This parameter indicates the channel pruning type: 1) `Filter`
            prune the filter of the convolution layer to prune the corresponding
            channels 2) `Batchnorm`: prune the channel in the batchnorm layer
        """
        self.prune_type = prune_type
        self.target_types = []
        if self.prune_type == 'Filter':
            self.target_types.extend(['Conv2d', 'Linear', 'ConvTranspose2d'])
        elif self.prune_type == "BatchNorm":
            self.target_types.append("BatchNorm2d")
        
        super(ChannelDependency, self).__init__(model, dummy_input, traced_model)

    def _get_parent_layers(self, node):
        """
        Find the nearest father conv layers for the target node.
        Parameters
        ---------
        node : torch._C.Node
            target node.
        Returns
        -------
        parent_layers: list
            nearest father conv/linear layers for the target worknode.
        """
        parent_layers = []
        queue = []
        queue.append(node)

        while queue:
            curnode = queue.pop(0)
            if curnode.op_type in self.target_types:
                parent_layers.append(curnode.name)
            elif curnode.op_type in RESHAPE_OPS:
                if reshape_break_channel_dependency(curnode):
                    continue
            parents = self.graph.find_predecessors(curnode.unique_name)
            parents = [self.graph.name_to_node[name] for name in parents]
            for parent in parents:
                queue.append(parent)
        return parent_layers

    def build_dependency(self):
        """
        Build the channel dependency for the conv layers
        in the model.
        """
        self.graph.unpack_manually()

        for node in self.graph.nodes_py.nodes_op:
            parent_layers = []

            if node.op_type in ADD_TYPES:
                parent_layers = self._get_parent_layers(node)
            
            elif node.op_type == CAT_TYPE:
                cat_dim = None
                for cnode in node.node_cpps:
                    cat_dim = list(cnode.inputs())[1].toIValue()
                    break
                if cat_dim != 1:
                    parent_layer = self._get_parent_layers(node)
            dependency_set = set(parent_layers)
            
            for parent in parent_layers:
                if parent in self.dependency:
                    dependency_set.update(self.dependency[parent])
                
            for _node in dependency_set:
                self.dependency[_node] = dependency_set
        
    def export(self, file_path):
        """
        export the channel dependencies as a csv file.
        The layers at the same line have output channel
        dependencies with each other. For example,
        layer1.1.conv2, conv1, and layer1.0.conv2 have
        output channel dependencies with each other, which
        means the output channel(filters) numbers of these
        three layers should be same with each other, otherwise
        the model may has shape conflict.
        Output example:
        Dependency Set,Convolutional Layers
        Set 1,layer1.1.conv2,layer1.0.conv2,conv1
        Set 2,layer1.0.conv1
        Set 3,layer1.1.conv1
        """
        header = ['Dependency Set', 'Layers']
        setid = 0
        visited = set()

        with open(file_path, 'w') as csvf:
            csv_w = csv.writer(csvf, delimiter=",")
            csv_w.writerow(header)

            for node in self.graph.nodes_py.nodes_op:
                if node.op_type not in self.target_types or node in visited:
                    continue
                setid += 1
                row = ['Set %d' %setid]
                if node.name not in self.dependency:
                    visited.add(node)
                    row.append(node.name)
                else:
                    for other in self.dependency[node.name]:
                        visited.add(self.graph.name_to_node[other])
                        row.append(other)
                
                csv_w.writerow(row)
    
    @property
    def dependency_set(self):
        """
        Get the list of the dependency set.

        Returns
        -------
        dependency_sets : list
            list of the dependency sets. For example,
            [set(['conv1', 'conv2']), set(['conv3', 'conv4'])]
        """
        d_sets = []
        visited = set()
        for node in self.graph.nodes_py.nodes_op:
            if node.op_type not in self.target_types or node in visited:
                continue
            tmp_set = set()
            if node.name not in self.dependency:
                visited.add(node)
                tmp_set.add(node.name)
            else:
                for other in self.dependency[node.name]:
                    visited.add(self.graph.name_to_node[other])
                    tmp_set.add(other)

                d_sets.append(tmp_set)
        return d_sets
        
class InputChannelDependency(ChannelDependency):
    """
    Some pruners may prune the input channel of the convolutional
    layers. While pruning the input channel of the convolutional layers,
    the layers that share the same input tensor should prune the same
    channels, and we say these layers that share the same input tensor/channel
    has the input channel dependency. If we only prune the input channel of one
    layer in the dependency set, there will be a shape conflict for the other
    layers in the same dependency set, which may trigger a runtime error.
    Here we judge whether the application will truncate the dependency by analyzing
    whether the number of channels before and after the operation has changed.
    If not, the input channel dependency will be passed to the following nodes.
    """
    def __init__(self, model, dummy_input, traced_model = None):
        """
        This model analyze the input channel dependencies between the conv
        layers in a model.
        Parameters
        ----------
        model : torch.nn.Module
            The model to be analyzed.
        data : torch.Tensor
            The example input data to trace the network architecture.
        traced_model : torch._C.Graph
            if we alreay has the traced graph of the target model, we donnot
            need to trace the model again.
        """
        super(InputChannelDependency, self).__init__(model, dummy_input, traced_model)
    
    def _get_following_convs(self, tensor):
        queue = []
        key_layers = []
        queue.extend(self.graph.input_to_node[tensor])
        while queue:
            curnode = queue.pop(0)
            if curnode.op_type == "Conv2d" or curnode.op_type == 'Linear' or curnode.op_type == "ConvTranspose2d":
                key_layers.append(curnode.name)
                continue
            
            elif curnode.op_type in RESHAPE_OPS:
                if reshape_break_channel_dependency(curnode):
                    continue
            
            successors = self.graph.find_successors(curnode.unique_name)
            successors = [self.graph.name_to_node[name] for name in successors]

            for layer in successors:
                queue.append(layer)
        return key_layers

    def build_dependency(self):
        """
        Build the input channel dependencies.
        The `InputChannelDependency` indicates the layers that have
        dependencies when pruning the input channel of the conv layers.
        In contrast, `ChannelDependency` indicates the dependent layers
        when pruning the output channles of conv layers (for example, L1FilterPruner).
        """
        self.graph.unpack_manually()
        for tensor in self.graph.input_to_node:
            layers = self._get_following_convs(tensor)
            dependency_set = set(layers)

            for layer in layers:
                if layer in self.dependency:
                    dependency_set.update(self.dependency[layer])
            for layer in dependency_set:
                self.dependency[layer] = dependency_set


class GroupDependency(Dependency):

    def __init__(self, model, dummy_input, traced_model =None):
        """
        This model analyze the group dependencis between the conv
        layers in a model.
        Parameters
        ----------
        model : torch.nn.Module
            The model to be analyzed.
        data : torch.Tensor
            The example input data to trace the network architecture.
        traced_model : torch._C.Graph
            if we alreay has the traced graph of the target model, we donnot
            need to trace the model again.
        """
        self.min_groups = {}
        super(GroupDependency, self).__init__(model, dummy_input, traced_model)

    def _get_parent_convs(self, node):
        """
        Find the nearest father conv layers for the target node.
        Parameters
        ---------
        node : torch._C.Node
            target node.
        Returns
        -------
        parent_layers : list
            nearest father conv layers for the target node. Due to the group
            dependency only exists between the conv layers, so we only find
            the parent conv layers.
        """
        parent_layers = []
        predecessors = self.graph.find_predecessors(node.unique_name)
        predecessors = [self.graph.node_to_name[node] for node in predecessors]

        queue = predecessors
        while queue:

            curnode = queue.pop(0)
            if curnode.op_type == "Conv2d" or curnode.optype == "ConvTranspose2d":
                parent_layers.append(curnode.name)
                continue
            parents = self.graph.find_predecessors(curnode.unique_name)
            parents = [self.graph.name_to_node[name] for name in parents]

            for parent in parents:
                queue.append(parent)
        return parent_layers

    
    def _get_conv_groups(self, node_group):
        """
        Get the number of groups for a convolutional layer.
        Parameters
        ----------
        node_group : NodePyGroup
            target node.
        Returns
        -------
        group : int
            the number of the groups of the target conv layer.
        """
        node_name = node_group.name
        _, leaf_module = get_module_by_name(self.model, node_name)
        if isinstance(leaf_module, PrunerModuleWrapper):
            leaf_module = leaf_module.module
        assert isinstance(leaf_module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
        group = leaf_module.groups
        n_filter = leaf_module.out_channels
        if n_filter == group:
            return 1
        return group

    def build_dependency(self):
        """
        Build the channel dependency for the conv layers
        in the model. This function return the group number
        of each conv layers. Note that, here, the group count
        of conv layers may be larger than their originl groups.
        This is because that the input channel will also be grouped
        for the group conv layers. To make this clear, assume we
        have two group conv layers: conv1(group=2), conv2(group=4).
        conv2 takes the output features of conv1 as input.
        Then we have to the filters of conv1 can still be
        divided into 4 groups after filter pruning, because
        the input channels of conv2 should be divided into
        4 groups.

        Returns
        -------
        self.dependency : dict
            key: the name of conv layers, value: the minimum value that the number of
            filters should be divisible to.
        """
        self.groups = {}
        for node in self.graph.nodes_py.nodes_op:
            if node.op_type == "Conv2d" or node.op_type == "ConvTranspose2d":
                group = self._get_conv_groups(node)
                if node.name in self.groups:
                    self.groups[node.name].append(group)
                else:
                    self.groups[node.name] = [group]
            
            if group > 1:
                parent_convs = self._get_parent_convs(node)
                for parent in parent_convs:
                    if parent in self.groups:
                        self.group[parent].append(group)
                    else:
                        self.groups[parent] = [group]
        
        for name in self.groups:
            self.dependency[name] = lcm_list(self.groups[name])
            if min(self.groups[name]) == gcd_list(self.groups[name]):
                self.min_groups[name] = min(self.groups[name])
            else:
                self.min_groups[name] = 1
        
        return self.dependency

    def export(self, file_path):
        """
        export the group dependency to a csv file.
        Each line describes a convolution layer, the
        first part of each line is the Pytorch module
        name of the conv layer. The second part of each
        line is the group count of the filters in this layer.
        Note that, the group count may be larger than this
        layers original group number.
        output example:
        Conv layer, Groups
        Conv1, 1
        Conv2, 2
        Conv3, 4
        """
        header = ['Conv Layer Name', 'Group']
        with open(file_path, 'w') as csvf:
            csv_w = csv.writer(csvf, delimiter = ",")
            csv_w.writerow(header)
            for name in self.dependency:
                group = self.dependency[name]
                csv_w.writerow([name, group])
    
    @property
    def dependency_set(self):
        return self.dependency


class ReshapeDependency(Dependency):

    def __init__(self, model = None, dummy_input = None, traced_model = None):
        """
        Some model may have the view/reshape functions, such functions may have fixed parameters
        and cannot be replaced at all. Therefore, these functions may have some constraints on
        their input shapes. In this class, we find the direct input conv/linear layers of these
        reshape functions. If you get the shape conflict when run the forward inference on the
        speeduped model, please try remove these layers from the pruner config list and try again.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be analyzed.
        data : torch.Tensor
            The example input data to trace the network architecture.
        traced_model : torch._C.Graph
            if we alreay has the traced graph of the target model, we donnot
            need to trace the model again.
        """
        super(ReshapeDependency, self).__init__(model, dummy_input, traced_model)

    def _get_parent_layers(self, node):
        parent_layers = []
        """
        Find the nearest father conv layers for the target node.

        Parameters
        ---------
        node : torch._C.Node
            target node.

        Returns
        -------
        parent_layers: list
            nearest father conv/linear layers for the target worknode.
        """
        queue = []
        queue.append(node)
        while queue:
            curnode = queue.pop(0)
            if curnode.op_type == "Conv2d" or curnode.op_type == "ConvTranspose2d" or curnode.op_type == "Linear":
                parent_layers.append(node.name)
                continue
            parents = self.graph.find_predecessors(curnode.unique_name)
            parents = [self.graph.name_to_node(name) for name in parents]
            for parent in parents:
                queue.append(parent) 
        return parent_layers

    def build_dependency(self):
        """
        Build the channel dependency for the conv layers
        in the model.
        """
        self.graph.unpack_manually()

        for node in self.graph.node_py.nodes_op:
            parent_layers = []
            for node.op_type in ['aten::view', 'aten::reshape']:
                _logger.info('Detect reshape-like functions: %s', node.op_type)
                parent_layers = self._get_parent_layers(node)
                print("Parent layers", parent_layers)
                self.dependency[node.unique_name] = parent_layers

    def export(self, file_path):
        """
        export the reshape dependencies as a csv file.

        Output example:
        Reshape OP, Dependent Layers
        model.view.1,layer1.1.conv2,layer1.0.conv2,conv1
        model.mean.1,layer1.0.conv1
        model.reshape.1,layer1.1.conv1
        """
        header = ['Reshape OP', 'Dependent Layers']
        with open(file_path, 'w') as csvf:
            csv_w = csv.writer(csvf, delimiter = ",")
            csv_w.writerow(header)

            for reshape_op in self.dependency:
                row = [reshape_op].extend(self.dependency[reshape_op])
                csv_w.writerow(row)
                

    @property
    def dependency_set(self):
        """
        Get the list of the dependency set.

        Returns
        -------
        dependency_sets : list
            list of the dependency sets. For example,
            [set(['conv1', 'conv2']), set(['conv3', 'conv4'])]

        """
        d_sets = []
        for rehsape_node in self.dependency:
            d_sets.extend(self.dependency[rehsape_node])
        d_sets = list(set(d_sets))
        return d_sets


class AttentionWeightDependency(Dependency):

    def __init__(self, model = None, dummy_input = None, traced_model = None):
        """
        Groups the linear layers belonging to the same attention layer in a model.
        Currently, we only capture weights in attention layers with forward computations written
        as four Linear layers (projections for Q, K, V, and output) and two matmul operations.
        The method implemented here can work for Huggingface transformers but may not correctly
        capture transformers written in other fashions (e.g., torch.nn.Transformer).

        Parameters
        ----------
        model : torch.nn.Module
            The model to be analyzed.
        dummy_input : torch.Tensor
            The example input data to trace the network architecture.
        traced_model : torch._C.Graph
            if we already have the traced graph of the target model, we do not
            need to trace the model again.
        """
        super( AttentionWeightDependency, self).__init__(model, dummy_input, traced_model)

    def _get_parent_layers(self, node):
        """
        Find the nearest parent linear layers for the target node.

        Parameters
        ---------
        node : torch._C.Node
            target node.

        Returns
        -------
        parent_layers: list
            nearest parent linear layers for the target worknode.
        """
        parent_layers = []
        queue = []
        queue.append(node)
        while queue:
            curnode = queue.pop(0)
            if curnode.op_type == "Linear":
                if curnode.name not in parent_layers:
                    parent_layers.append(curnode.name)
                continue
            if curnode.op_type == "LayerNorm":
                continue

            parents = self.graph.find_predecessors(curnode.unique_name)
            parents = [self.graph.name_to_node[name] for name in parents]
            for parent in parents:
                queue.append(parent)
        return parent_layers


    def _get_children_layers(self, node):
        """
        Find the nearest children linear layers for the target node.

        Parameters
        ---------
        node : torch._C.Node
            target node.

        Returns
        -------
        children_layers: list
            nearest children linear layers for the target worknode.
        """
        children_layers = []
        queue = []
        queue.append(node)
        while queue:
            curnode = queue.pop(0)
            if curnode.op_type == "Linear":
                if curnode.name not in children_layers:
                    children_layers.append(curnode.name)
                continue
            if curnode.op_type == "LayerNorm":
                continue
            children = self.graph.find_successors(curnode.unique_name)
            children = [self.graph.name_to_node[name] for name in children]
            for child in children:
                queue.append(children)
        return children_layers

    def build_dependency(self):
        """
        For every matmul operation, find the immediate parent and children Linear operations.
        If we get three parents and one children, add these four weights as a dependecy group.
        """
        self.graph.unpack_manually()

        for node in self.graph.nodes_py.nodes_op:
            layers = []
            if node.op_type == "aten::matmul":
                parent_layers = self._get_parent_layers(node)
                children_layers = self._get_children_layers(node)

                if len(parent_layers) == 3 and len(children_layers) == 1:
                    layers.extend(parent_layers)
                    layers.extend(children_layers)

            self.dependency[node.name] = layers

    @property
    def dependency_set(self):
        """
        Get the list of the dependency set.

        Returns
        -------
        dependency_sets : list
            list of the dependency sets.
            Each dependency set is a 4-element list of module names, with the first three elements being the projection
            matrices for Q, K, V (in any order), and the last element being the dense matrix.
        """
        d_sets = []
        for node in self.graph.nodes_py.nodes_op:
            if node.op_type != "aten::matmul" or node.name not in self.dependency or len(self.dependency[node.name]) != 4:
                continue
            d_sets.append(self.dependency[node.name])
        return d_sets

    def export(self, file_path):
        """
        Export the group dependency to a csv file. Each line describes an attention layer.

        Output example:
        Attention layer matmul op, Group
        """
        header = ['Attentiobn layer matmul op', 'Group']
        with open(file_path, 'w') as csvf:
            csv_w = csv.writer(csvf, delimiter = ",")
            csv_w.writerow(header)

            for name in self.dependency:
                group = self.dependency[name]
                if len(group) > 0:
                    csv_w.writerow([name, group])



