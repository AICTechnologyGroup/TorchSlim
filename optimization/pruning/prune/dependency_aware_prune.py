import logging
from schema import And, Optional
from optimization.common.graph import TorchModuleGraph
from optimization.pruning.core import ChannelDependency, GroupDependency
from optimization.common.base import PrunerSchema
from optimization.pruning.core import Pruner
from optimization.pruning.core import SlimPrunerMasker, L1FilterPrunerMasker, L2FilterPrunerMasker, FPGMPrunerMasker, TaylorFOWeightFilterPrunerMasker, ActivationAPoZRankFilterPrunerMasker, ActivationMeanRankFilterPrunerMasker


MASKER_DICT = {
    'slim': SlimPrunerMasker,
    'l1': L1FilterPrunerMasker,
    'l2': L2FilterPrunerMasker,
    'fpgm': FPGMPrunerMasker,
    'taylorfo': TaylorFOWeightFilterPrunerMasker,
    'apoz': ActivationAPoZRankFilterPrunerMasker,
    'mean_activation': ActivationMeanRankFilterPrunerMasker
}

__all__ = ['DependencyAwarePruner']

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class DependencyAwarePruner(Pruner):
    """
    DependencyAwarePruner has two ways to calculate the masks
    for conv layers. In the normal way, the DependencyAwarePruner
    will calculate the mask of each layer separately. For example, each
    conv layer determine which filters should be pruned according to its L1
    norm. In constrast, in the dependency-aware way, the layers that in a
    dependency group will be pruned jointly and these layers will be forced
    to prune the same channels.
    """
    def __init__(self, model, config_list, optimizer=None, pruning_algorithm='level', dependency_aware=False,
                 dummy_input=None, **algo_kwargs):
        super().__init__(model, config_list=config_list, optimizer=optimizer)

        self.dependency_aware = dependency_aware
        self.dummy_input = dummy_input

        if self.dependency_aware:
            if not self._supported_dependency_aware():
                raise ValueError('This pruner does not support dependency aware!')

            errmsg = "When dependency_aware is set, the dummy_input should not be None"
            assert self.dummy_input is not None, errmsg

            self._unwrap_model()
            self.graph = TorchModuleGraph(model, dummy_input)
            self._wrap_model()
            self.channel_depen = ChannelDependency(model, dummy_input, traced_model=self.graph.trace)
            self.group_depen = GroupDependency(model, dummy_input, traced_model=self.graph.trace)
            self.channel_depen = self.channel_depen.dependency_sets
            self.channel_depen = {
                name: sets for sets in self.channel_depen for name in sets}
            self.group_depen = self.group_depen.dependency_sets

        self.masker = MASKER_DICT[pruning_algorithm](
            model, self, **algo_kwargs)
        self.masker.dependency_aware = dependency_aware
        self.set_wrappers_attribute("if_calculated", False)

    def calc_mask(self, wrapper, wrapper_idx=None):
        if not wrapper.if_calculated:
            sparsity = wrapper.config['sparsity']
            masks = self.masker.calc_mask(
                sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)

            if masks is not None:
                wrapper.if_calculated = True
            return masks
        else:
            return None

    def update_mask(self):
        if not self.dependency_aware:
            super(DependencyAwarePruner, self).update_mask()
        else:
            self._dependency_update_mask()

    def validate_config(self, model, config_list):
        schema = PrunerSchema([{
            Optional('sparsity'): And(float, lambda n: 0 < n < 1),
            Optional('op_types'): ['Conv2d'],
            Optional('op_names'): [str],
            Optional('exclude'): bool
        }], model, _logger)

        schema.validate(config_list)

    def _supported_dependency_aware(self):
        raise NotImplementedError

    def _dependency_calc_mask(self, wrappers, channel_dsets, wrappers_idx=None):
        """
        calculate the masks for the conv layers in the same
        channel dependecy set. All the layers passed in have
        the same number of channels.

        Parameters
        ----------
        wrappers: list
            The list of the wrappers that in the same channel dependency
            set.
        wrappers_idx: list
            The list of the indexes of wrapppers.
        Returns
        -------
        masks: dict
            A dict object that contains the masks of the layers in this
            dependency group, the key is the name of the convolutional layers.
        """
        groups = [self.group_depen[_w.name] for _w in wrappers]
        sparsities = [_w.config['sparsity'] for _w in wrappers]
        masks = self.masker.calc_mask(
            sparsities, wrappers, wrappers_idx, channel_dsets=channel_dsets, groups=groups)
        if masks is not None:
            for _w in wrappers:
                _w.if_calculated = True
        return masks

    def _dependency_update_mask(self):
        """
        In the original update_mask, the wraper of each layer will update its
        own mask according to the sparsity specified in the config_list. However, in
        the _dependency_update_mask, we may prune several layers at the same
        time according the sparsities and the channel/group dependencies.
        """
        name2wrapper = {x.name: x for x in self.get_modules_wrapper()}
        wrapper2index = {x: i for i, x in enumerate(self.get_modules_wrapper())}
        for wrapper in self.get_modules_wrapper():
            if wrapper.if_calculated:
                continue
            _names = [x for x in self.channel_depen[wrapper.name]]
            _logger.info('Pruning the dependent layers: %s', ','.join(_names))
            _wrappers = [name2wrapper[name]
                         for name in _names if name in name2wrapper]
            _wrapper_idxes = [wrapper2index[_w] for _w in _wrappers]

            masks = self._dependency_calc_mask(
                _wrappers, _names, wrappers_idx=_wrapper_idxes)
            if masks is not None:
                for layer in masks:
                    for mask_type in masks[layer]:
                        assert hasattr(name2wrapper[layer], mask_type), "there is no attribute '%s' in wrapper on %s" \
                            % (mask_type, layer)
                        setattr(name2wrapper[layer], mask_type, masks[layer][mask_type])
