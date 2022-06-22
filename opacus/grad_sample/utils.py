#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Type, Union

import torch
import torch.nn as nn

from .grad_sample_module import GradSampleModule
from .gsm_base import AbstractGradSampleModule


def register_grad_sampler(
    target_class_or_classes: Union[Type[nn.Module], Sequence[Type[nn.Module]]]
):
    """
    Registers the decorated function as the ``grad_sampler`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to compute a per-sample gradient
    of ``target_class_or_classes``. The signature of every grad_sampler is always the same:

    >>> @register_grad_sampler(MyCustomModel)
    ... def compute_grad_sample(module, activations, backprops):
    ...    pass

    It may help you to take a look at the existing grad_samplers inside Opacus, under ``opacus.grad_sample.``
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            GradSampleModule.GRAD_SAMPLERS[target_class] = f
        return f

    return decorator


def wrap_model(model: nn.Module, grad_sample_mode: str, *args, **kwargs):
    cls = get_gsm_class(grad_sample_mode)
    return cls(model, *args, **kwargs)


def get_gsm_class(grad_sample_mode: str) -> Type[AbstractGradSampleModule]:
    """
    Returns AbstractGradSampleModule subclass correspinding to the input mode.
    See README for detailed comparison between grad sample modes.

    :param grad_sample_mode:
    :return:
    """
    if grad_sample_mode == "hooks":
        return GradSampleModule
    elif grad_sample_mode == "ew":
        try:
            from opacus.grad_sample.gsm_exp_weights import (
                GradSampleModuleExpandedWeights,
            )

            return GradSampleModuleExpandedWeights
        except ImportError:
            raise ImportError(
                f"Requested grad_sample_mode=ew, "
                f"but found PyTorch version={torch.__version__}. "
                f"ExpandedWeights available for torch>=1.12. "
                f"Please install recent PyTorch or use grad_sample_mode=hooks"
            )
    else:
        raise ValueError(
            f"Unexpected grad_sample_mode: {grad_sample_mode}. "
            f"Allowed values: hooks, ew"
        )
