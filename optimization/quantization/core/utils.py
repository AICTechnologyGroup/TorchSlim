from enum import Enum, EnumMeta
from typing import Any, Optional
import torch


class _QuantLiteralEnumMeta(EnumMeta):

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True

class _QuantLiteralEnum(Enum, metaclass = _QuantLiteralEnumMeta):
    pass

class QuantScheme(str, _QuantLiteralEnum):
    PER_TENSOR_AFFINE = 'per_tensor_affine'
    PER_TENSOR_SYMMETRIC = 'per_tensor_symmetric'
    PER_CHANNEL_AFFINE = 'per_channel_affine'
    PER_CHANNEL_SYMMETRIC = 'per_channel_symmetric'


PER_CHANNEL_QUANT_SCHEME = [QuantScheme.PER_CHANNEL_AFFINE, QuantScheme.PER_CHANNEL_SYMMETRIC]


class QuantDtype(str, _QuantLiteralEnum):
    UINT = 'uint'
    INT = 'int'


class QuantType(str, _QuantLiteralEnum):
    INPUT = 'input'
    WEIGHT = 'weight'
    OUTPUT = 'output'

    def type_to_scale_zero_point_name(self):
        if self == QuantType.INPUT:
            return 'input_scale', 'input_zero_point'
        elif self == QuantType.WEIGHT:
            return 'weight_scale', 'weight_zero_point'
        elif self == QuantType.OUTPUT:
            return 'output_scale', 'output_zero_point'
        else:
            raise TypeError


class QuantConfigLiteral(str, _QuantLiteralEnum):
    QUANT_SETTINGS = 'quant_settings'
    QUANT_SCHEME = 'quant_scheme'
    QUANT_DTYPE = 'quant_dtype'
    BITS = 'bits'
    QMIN = 'qmin'
    QMAX = 'qmax'
    INPUT_SCALE = 'input_scale'
    INPUT_ZERO_POINT = 'input_zero_point'
    OUTPUT_SCALE = 'output_scale'
    OUTPUT_ZERO_POINT = 'output_zero_point'
    WEIGHT_SCALE = 'weight_scale'
    WEIGHT_ZERO_POINT = 'weight_zero_point'


BN_FOLD_OP = ["Conv2d"]
BN_FOLD_TAG = 'BN_FOLD_TAG'


quant_default_settings = {
    QuantType.WEIGHT: {
        'quant_scheme': QuantScheme.PER_TENSOR_AFFINE,
        'quant_dtype': QuantDtype.UINT,
    },
    QuantType.INPUT: {
        'quant_scheme': QuantScheme.PER_TENSOR_AFFINE,
        'quant_dtype': QuantDtype.UINT
    },
    QuantType.OUTPUT: {
        'quant_scheme': QuantScheme.PER_TENSOR_AFFINE,
        'quant_dtype': QuantDtype.UINT
    }
}


def calculate_qmin_qmax(bits, dtype):
    if dtype == QuantDtype.INT:
        qmin, qmax = -2 ** (bits - 1) + 1, 2 ** (bits - 1) - 1
    elif dtype == QuantDtype.UINT:
        qmin, qmax = 0, 2 ** (bits) - 1
    else:
        raise TypeError("Wrong quantization dtype, please make sure it is one of 'int' and 'uint'.")
    return qmin, qmax

def get_bits_length(config, quant_type):
    if isinstance(config['quant_bits'], int):
        return config['quant_bits']
    else:
        return config['quant_bits'].get(quant_type)

def is_per_channel(quant_scheme):
    if quant_scheme in [QuantScheme.PER_CHANNEL_AFFINE, QuantScheme.PER_CHANNEL_SYMMETRIC]:
        return True
    return False

def get_quant_shape(shape, quant_type, quant_scheme):
    default_idx = 0 if quant_type == QuantType.WEIGHT else 1
    if is_per_channel(quant_scheme):
        quant_shape = [1 if idx != default_idx else s for idx, s in enumerate(shape)]
    else:
        quant_shape = [1]
    return quant_shape

def get_target_dim(quant_type, quant_scheme):
    default_idx = 0 if quant_type == QuantType.WEIGHT else 1 
    if is_per_channel(quant_scheme):
        return default_idx
    else:
        return None


def get_min_max_value(x, quant_type, quant_scheme):
    target_dim = get_target_dim(quant_type, quant_scheme)
    if target_dim is None:
        return torch.min(x), torch.max(x)
    
    indices = list(range(len(x.shape)))
    assert target_dim < len(indices), "target_dim needs to be less than the number of dim of the tensor"
    del indices[target_dim]

    min_val = torch.amin(x, indices, keepdims = True)
    max_val = torch.amax(x, indices, keepdims = True)

    return min_val, max_val

class TensorQuantSetting:

    def __init__(self, **kwargs):

        self._fields = {}
        for k, v in kwargs.items():
            self._fields[k] = v

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self._fields[name] = val

    def __getattr__(self, name):
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find {} in TensorQuantSetting!".format(name))
        return self._fields[name]

    def get_qmin_qmax(self):
        assert 'qmin' in self._fields and 'qmax' in self._fields, "Can not found qmin & qmax in TensorQuantSetting"
        return self._fields['qmin'], self._fields['qmax']



class LayerQuantSetting(object):
    def __init__(self, config):
        self.input: Optional[TensorQuantSetting] = None
        self.weight: Optional[TensorQuantSetting] = None
        self.output: Optional[TensorQuantSetting] = None
        self._extra_layer_setting = {}

        for quant_type in QuantType:
            if quant_type in config.get("quant_types", []):
                setting = TensorQuantSetting()

                quant_scheme = self.parse_optional_config(config, quant_type, 'quant_scheme')
                setting.quant_scheme = quant_scheme
                quant_dtype = self.parse_optional_config(config, quant_type, 'quant_dtype')
                setting.quant_dtype = quant_dtype

                bits = get_bits_length(config, quant_type)
                qmin, qmax = calculate_qmin_qmax(bits, quant_dtype)
                setting.bits = bits
                setting.qmin = qmin
                setting.qmax = qmax
                setattr(self, quant_type, setting)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_") or name in QuantType:
            super().__setattr__(name, val)
        else:
            self._extra_layer_setting[name] = val

    def __getattr__(self, name):
        if name == "_extra_layer_setting" or name not in self._extra_layer_setting:
            raise AttributeError("Cannot find {} in LayerQuantSetting!".format(name))
        return self._extra_layer_setting[name]

    @staticmethod
    def parse_optional_config(config, quant_type, target):
        def get_config(config, quant_type, target):
            if not config.get(target):
                return None

            if isinstance(config[target], dict):
                return config[target].get(quant_type)
            else:
                return config[target]

        default_val = quant_default_settings[quant_type].get(target, None)
        config_val = get_config(config, quant_type, target)
        val = config_val if config_val else default_val
        return val