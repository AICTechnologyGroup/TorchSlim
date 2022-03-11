import copy
import types
import logging
import torch
from torch.autograd.grad_mode import F
from schema import Schema, And, SchemaError


_logger = logging.getLogger(__name__)



weighted_modules = [
    'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'Linear', 'Bilinear',
    'PReLU',
    'Embedding', 'EmbeddingBag',
]


class LayerInfo:

    def __init__(self, name, module):
        self.module = module
        self.name = name
        self.type = type(module).__name__


def _setattr(model, name, module):
    name_list = name.split('.')
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)


class Compressor:

    def __init__(self, model, config_list, optimizer=None):
        """
        Record necessary info in class members

        Parameters
        ----------
        model : pytorch model
            the model user wants to compress
        config_list : list
            the configurations that users specify for compression
        optimizer: pytorch optimizer
            optimizer used to train the model
        """

        assert isinstance(model, torch.nn.Module)

        self.validate_config(model, config_list)

        self.bound_model = model
        self.config_list = config_list
        self.optimizer = optimizer
        self.modules_to_compress = None
        self.modules_wrapper = []
        self.is_wrapped = False
        self._fwd_hook_handles= {}
        self._fwd_hook_id = 0
        self.reset()

        if not self.modules_wrapper:
            _logger.warning("Nothing is configured to compress, please check your model and config_list")

    def validate_config(self, model, config_list):
        """
        subclass can optionally implement this method to check if config_list if valid
        """
        pass


    def reset(self, checkpoint = None):
        """
        reset model state dict and model wrapper
        """
        self._unwarp_model()
        if checkpoint is not None:
            self.bound_model.load_state_dict(checkpoint)
        
        self.modules_to_compress = None
        self.modules_wrapper = []

        for layer, config in self._detect_modules_to_compress():
            wrapper = self._warp_modules(layer, config)
            self.modules_wrapper.append(wrapper)

        self._warp_model()

    def _detect_modules_to_compress(self):
        """
        detect all modules should be compressed, and save the result in `self.modules_to_compress`.
        The model will be instrumented and user should never edit it after calling this method.
        """
        if self.modules_to_compress is None:
            self.modules_to_compress = []
            for name, module in self.bound_model.named_modules():
                if module == self.bound_model:
                    continue
                layer = LayerInfo(name, module)
                config = self.select_config(layer)
                if config is not None:
                    self.modules_to_compress.append((layer, config))
        return self.modules_to_compress


    def _warp_model(self):
        """
        wrap all modules that needed to be compressed

        """
        for wrapper in self.get_modules_wrapper():
            _setattr(self.bound_model, wrapper.name, wrapper.module)
        self.is_wrapped = False

    
    def _unwarp_model(self):
        """
        unwrap all modules that needed to be compressed

        """
        for wrapper in self.get_modules_wrapper():
            _setattr(self.bound_model, wrapper.name, wrapper.module)
        self.is_wrapped = False
    
    def compress(self):
        """
        Compress the model with algorithm implemented by subclass.

        The model will be instrumented and user should never edit it after calling this method.
        `self.modules_to_compress` records all the to-be-compressed layers

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """        
        return self.bound_model

    def set_wrappers_attribute(self, name, value):

        """
        To register attributes used in wrapped module's forward method.
        If the type of the value is Torch.tensor, then this value is registered as a buffer in wrapper,
        which will be saved by model.state_dict. Otherwise, this value is just a regular variable in wrapper.

        Parameters
        ----------
        name : str
            name of the variable
        value: any
            value of the variable
        """

        for wrapper in self.get_modules_wrapper():
            if isinstance(value, torch.Tensor):
                wrapper.register_buffer(name, value.clone())
            else:
                setattr(wrapper, name, value)
    
    def get_modules_to_compress(self):
        """
        To obtain all the to-be-compressed modules.

        Returns
        -------
        list
            a list of the layers, each of which is a tuple (`layer`, `config`),
            `layer` is `LayerInfo`, `config` is a `dict`
        """
        return self.modules_to_compress

    def get_modules_wrapper(self):
        """
        To obtain all the wrapped modules.

        Returns
        -------
        list
            a list of the wrapped modules
        """
        return self.modules_wrapper

    def select_config(self, layer):
        """
        Find the configuration for `layer` by parsing `self.config_list`

        Parameters
        ----------
        layer : LayerInfo
            one layer

        Returns
        -------
        config or None
            the retrieved configuration for this layer, if None, this layer should
            not be compressed
        """
        ret = None
        for config in self.config_list:
            config = config.copy()
            if 'op_types' in config and "default" in config ['op_types']:
                expanded_op_types = []
                for op_type in config['op_types']:
                    if op_type == "default":
                        expanded_op_types.extend(weighted_modules)
                    else:
                        expanded_op_types.append(op_type)
                config['op_types'] = expanded_op_types

            if 'op_types' in config and layer.type not in config['op_types']:
                continue
            
            if "op_names" in config and layer.name not in config['op_names']:
                continue

            ret = config
        if ret is None or 'exclude' in ret:
            return None
        return ret
    
    def update_epoch(self, epoch):
        """
        If user want to update model every epoch, user can override this method.
        This method should be called at the beginning of each epoch

        Parameters
        ----------
        epoch : num
            the current epoch number
        """
        pass

    def _warp_modules(self, layer, config):
        """
        This method is implemented in the subclasses, i.e., `Pruner` and `Quantizer`

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            the configuration for compressing this layer
        """
        raise NotImplementedError()

    def add_activation_collector(self, collector):
        self._fwd_hook_id += 1
        self._fwd_hook_handles[self._fwd_hook_id] = []
        for wrapper in self.get_modules_wrapper():
            handle = wrapper.register_forward_hook(collector)
            self._fwd_hook_handles[self._fwd_hook_id].append(handle)
        return self._fwd_hook_id

    def remove_activation_collector(self, fwd_hook_id):
        if fwd_hook_id not in self._fwd_hook_handles:
            raise ValueError("%s is not a valid collector id" % str(fwd_hook_id))
        for handle in self._fwd_hook_handles[fwd_hook_id]:
            handle.remove()
        del self._fwd_hook_handles[fwd_hook_id]

    def patch_optimizer(self, *tasks):
        def patch_step(old_step):
            def new_step(_, *args, **kwargs):
                output = old_step(*args, **kwargs)

                for task in tasks:
                    task()
                return output
            return new_step
        if self.optimizer is not None:
            self.optimizer.step = types.MethodType(patch_step(self.optimizer.step), self.optimizer)

    def patch_optimizer_before(self, *tasks):
        def patch_step(old_step):
            def new_step(_, *args, **kwargs):
                for task in tasks:
                    task()
                output = old_step(*args, **kwargs)
                return output
            return new_step
        if self.optimizer is not None:
            self.optimizer.step = types.MethodType(patch_step(self.optimizer.step), self.optimizer)
    


def validate_op_names(model, op_names, logger):
    found_names = set(map(lambda x: x[0], model.named_modules()))

    not_found_op_names = list(set(op_names) - found_names)
    if not_found_op_names:
        logger.warning('op_names %s not found in model', not_found_op_names)

    return True

def validate_op_types(model, op_types, logger):
    found_types = set(['default']) | set(map(lambda x: type(x[1]).__name__, model.named_modules()))

    not_found_op_types = list(set(op_types) - found_types)
    if not_found_op_types:
        logger.warning('op_types %s not found in model', not_found_op_types)

    return True

def validate_op_types_op_names(data):
    if not ('op_types' in data or 'op_names' in data):
        raise SchemaError('Either op_types or op_names must be specified.')
    return True


class CompressorSchema:
    def __init__(self, data_schema, model, logger):
        assert isinstance(data_schema, list) and len(data_schema) <= 1
        self.data_schema = data_schema
        self.compressor_schema = Schema(self._modify_schema(data_schema, model, logger))

    def _modify_schema(self, data_schema, model, logger):
        if not data_schema:
            return data_schema

        for k in data_schema[0]:
            old_schema = data_schema[0][k]
            if k == 'op_types' or (isinstance(k, Schema) and k._schema == 'op_types'):
                new_schema = And(old_schema, lambda n: validate_op_types(model, n, logger))
                data_schema[0][k] = new_schema
            if k == 'op_names' or (isinstance(k, Schema) and k._schema == 'op_names'):
                new_schema = And(old_schema, lambda n: validate_op_names(model, n, logger))
                data_schema[0][k] = new_schema

        data_schema[0] = And(data_schema[0], lambda d: validate_op_types_op_names(d))

        return data_schema

    def validate(self, data):
        self.compressor_schema.validate(data)

def validate_exclude_sparsity(data):
    if not ('exclude' in data or 'sparsity' in data):
        raise SchemaError('Either sparisty or exclude must be specified.')
    return True

def validate_exclude_quant_types_quant_bits(data):
    if not ('exclude' in data or ('quant_types' in data and 'quant_bits' in data)):
        raise SchemaError('Either (quant_types and quant_bits) or exclude must be specified.')
    return True

class PrunerSchema(CompressorSchema):
    def _modify_schema(self, data_schema, model, logger):
        data_schema = super()._modify_schema(data_schema, model, logger)
        data_schema[0] = And(data_schema[0], lambda d: validate_exclude_sparsity(d))
        return data_schema

class QuantizerSchema(CompressorSchema):
    def _modify_schema(self, data_schema, model, logger):
        data_schema = super()._modify_schema(data_schema, model, logger)
        data_schema[0] = And(data_schema[0], lambda d: validate_exclude_quant_types_quant_bits(d))
        return data_schema


