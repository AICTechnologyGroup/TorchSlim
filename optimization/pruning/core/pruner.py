import torch
from optimization.common.base import Compressor
import logging

_logger = logging.getLogger(__name__)

class PrunerModuleWrapper(torch.nn.Module):

    def __init__(self, module, module_name, module_type, config, pruner):
        """
        Wrap an module to enable data parallel, forward method customization and buffer registeration.

        Parameters
        ----------
        module : pytorch module
            the module user wants to compress
        config : dict
            the configurations that users specify for compression
        module_name : str
            the name of the module to compress, wrapper module shares same name
        module_type : str
            the type of the module to compress
        pruner ： Pruner
            the pruner used to calculate mask
        """
        super().__init__()
        self.module = module
        self.name = module_name
        self.type = module_type
        self.config = config
        self.pruner = pruner


        self.register_buffer("weight_mask", torch.ones(self.module.weight.shape))
        if hasattr(self.module, "bias") and self.module.bias is not None:
            self.register_buffer("bias_mask", torch.ones(self.module.bias.shape))
        else:
            self.register_buffer("bias_mask", None)

    def forward(self, *input):
        self.module.weight.data = self.module.weight.data.mul_(self.weight_mask)
        if hasattr(self.module, "bias") and self.module.bias is not None:
            self.module.bias.data = self.module.bias.data.mul_(self.bias_mask)
        return self.module(*input)

class Pruner(Compressor):
    """
    Prune to an exact pruning level specification

    Attributes
    ----------
    mask_dict : dict
        Dictionary for saving masks, `key` should be layer name and
        `value` should be a tensor which has the same shape with layer's weight

    """
    def __init__(self, model, config_list, optimizer = None):
        super().__init__(model, config_list, optimizer)
    
    def compress(self):
        self.update_mask()
        return self.bound_model
    
    def upadte_mask(self):
        for wrapper_idx, wrapper in enumerate(self.get_modules_wrapper()):
            masks = self.calc_mask(wrapper, wrapper_idx = wrapper_idx)
            if masks is not None:
                for k in masks:
                    assert hasattr(wrapper, k), "there is no attribute '%s' in wrapper" %k
                    setattr(wrapper, k, masks[k])


    def calc_mask(self, wrapper, **kwargs):
        """
        Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with `mul()` operation on the weight.
        This method is effectively hooked to `forward()` method of the model.

        Parameters
        ----------
        wrapper : Module
            calculate mask for `wrapper.module`'s weight
        """
        raise NotImplementedError("Pruners must overload calc_mask()")

    def _wrap_modules(self, layer, config):
        """
        Create a wrapper module to replace the original one.

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config : dict
            the configuration for generating the mask
        """
        _logger.debug("Module detected to compress: %s.", layer.name)
        wrapper = PrunerModuleWrapper(layer.module, layer.name, layer.type, config, self)
        assert hasattr(layer.module, 'weight'), "module %s does not have 'weight' attribute" % layer.name
        wrapper.to(layer.module.weight.device)
        return wrapper

    def export_model(self, model_path, mask_path = None, onnx_path = None, input_shape = None, device =None, dummy_input = None, opset_version = None):
        """
        Export pruned model weights, masks and onnx model(optional)

        Parameters
        ----------
        model_path : str
            path to save pruned model state_dict
        mask_path : str
            (optional) path to save mask dict
        onnx_path : str
            (optional) path to save onnx model
        input_shape : list or tuple
            input shape to onnx model, used for creating a dummy input tensor for torch.onnx.export
            if the input has a complex structure (e.g., a tuple), please directly create the input and
            pass it to dummy_input instead
            note: this argument is deprecated and will be removed; please use dummy_input instead
        device : torch.device
            device of the model, where to place the dummy input tensor for exporting onnx file;
            the tensor is placed on cpu if ```device``` is None
            only useful when both onnx_path and input_shape are passed
            note: this argument is deprecated and will be removed; please use dummy_input instead
        dummy_input: torch.Tensor or tuple
            dummy input to the onnx model; used when input_shape is not enough to specify dummy input
            user should ensure that the dummy_input is on the same device as the model
        opset_version: int
            opset_version parameter for torch.onnx.export; only useful when onnx_path is not None
            if not passed, torch.onnx.export will use its default opset_version
        """
        assert model_path is not None, "model_path must be specified"
        mask_dict = {}
        self._unwarp_model()

        for wrapper in self.get_modules_wrapper():
            weight_mask = wrapper.weight_mask
            bias_mask = wrapper.bias_mask
            if weight_mask is not None:
                mask_sum = weight_mask.sum().item()
                mask_num = weight_mask.numel()
                _logger.debug("Layer: %s Sparsity: %.4f", wrapper.name, 1 - mask_sum / mask_num)
                wrapper.module.weight.data = wrapper.module.weight.data.mul(weight_mask)
            if bias_mask is not None:
                wrapper.module.bias.data = wrapper.module.bias.data.mul(bias_mask)
            mask_dict[wrapper.name] = {
                "weight" : weight_mask,
                "bias" : bias_mask
            }
        torch.save(self.bound_model.state_dict(), model_path)
        _logger.info("Model state_dict saved to %s", model_path)

        if mask_path is not None:
            torch.save(mask_dict, mask_path)
        
        if onnx_path is not None:
            assert input_shape is not None or dummy_input is not None, 'input_shape or dummy_input must be specified to export onnx model'

            if dummy_input is None:
                _logger.warning("""The argument input_shape and device will be removed in the future. Please create a dummy input and pass it to dummy_input instead.""")

                if device is None:
                    device = torch.device("cpu")
                input_data = torch.Tensor(*input_shape).to(device)
            else:
                input_data = dummy_input
            if opset_version is not None:
                torch.onnx.export(self.bound_model, input_data, onnx_path, opset_version = opset_version)
            else:
                torch.onnx.export(self.bound_model, input_data, onnx_path)

            if dummy_input is None:
                _logger.info("Model in onnx with input shape %s saved to %s", input_data.shape, onnx_path)
            else:
                _logger.info("Model in onxx saved to %s", onnx_path)
        self._warp_model()

    def load_model_state_dict(self, model_state):
        """
        Load the state dict saved from unwrapped model.

        Parameters
        ----------
        model_state : dict
            state dict saved from unwrapped model
        """
        if self.is_wrapped:
            self._unwarp_model()
            self.bound_model.load_state_dict(model_state)
            self._warp_model()
        else:
            self.bound_model.load_state_dict(model_state)

    def get_pruned_weights(self, dim = 0):
        """
        Log the simulated prune sparsity.

        Parameters
        ----------
        dim : int
            the pruned dim.
        """
        for _, wrapper in enumerate(self.get_modules_wrapper):
            weight_mask = wrapper.weight_mask
            mask_size = weight_mask.size()
            if len(mask_size) == 1:
                index = torch.nonzero(weight_mask.abs() != 0).tolist()
            else:
                sum_idx = list(range(len(mask_size)))
                sum_idx.remove(dim)
                index = torch.nonzero(weight_mask.abs().sum(sum_idx) != 0).tolist()

            _logger.info(f"simulated prune {wrapper.name} remain/total : {len(index)}/{weight_mask.size(dim)}")
    