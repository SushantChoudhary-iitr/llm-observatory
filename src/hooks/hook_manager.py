import torch
from typing import Dict, Any
from src.profilers.timer import Timer


class HookManager:

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.records = []
        self.timer = Timer(model.device)
        #self.start_times = {}

    # def register_hooks(self):
    #     for name, module in self.model.named_modules():

    #         if self._should_instrument(name, module):
    #             hook = module.register_forward_hook(
    #                 self._create_hook(name)
    #             )

    #             self.hooks.append(hook)

    def register_hooks(self):

        for name, module in self.model.named_modules():

            if self._should_instrument(name, module): # _should_instrument is like a filter functio, return true for specific shit

                pre_hook = module.register_forward_pre_hook(
                    self._create_pre_hook(name)
                )

                post_hook = module.register_forward_hook(
                    self._create_post_hook(name)
                )

                self.hooks.append(pre_hook)
                self.hooks.append(post_hook)

    def _create_pre_hook(self, name):

        def hook(module, inputs):

            self.timer.start()
            #self.start_times[name] = self.timer.start_time

        return hook

    def _create_post_hook(self, name):

        def hook(module, inputs, output):

            latency = self.timer.stop()

            record = {
                "layer_name": name,
                "output_shape": self._get_output_shape(output),
                "latency_ms": latency
            }

            self.records.append(record)

        return hook

#Looking for Attention and MLP
    def _should_instrument(self, name, module):
        return (
            not name.endswith("_lin") 
            and
            (".attention" in name
            or name.endswith(".ffn"))
        )

    def _create_hook(self, name):

        def hook(module, inputs, output):

            record = {
                "layer_name": name,
                "output_shape": self._get_output_shape(output)
            }

            self.records.append(record)

        return hook

    def get_records(self):
        return self.records.copy()

    def _get_output_shape(self, output):

        if isinstance(output, torch.Tensor):
            return tuple(output.shape)

        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return tuple(item.shape)

        return None

    def remove_hooks(self):

        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()