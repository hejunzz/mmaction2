# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .distribution_env import build_ddp, build_dp, default_device
from .gradcam_utils import GradCAM
from .logger import get_root_logger
from .logging import get_logger, print_log
from .misc import get_random_string, get_shm_dir, get_thread_id
from .module_hooks import register_module_hooks
# from .precise_bn import PreciseBNHook
from .setup_env import setup_multi_processes

from .parrots_wrapper import (IS_CUDA_AVAILABLE, TORCH_VERSION,
                                BuildExtension, CppExtension, CUDAExtension,
                                DataLoader, PoolDataLoader, SyncBatchNorm,
                                _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd,
                                _AvgPoolNd, _BatchNorm, _ConvNd,
                                _ConvTransposeMixin, _get_cuda_home,
                                _InstanceNorm, _MaxPoolNd, get_build_config,
                                is_rocm_pytorch)

__all__ = [
    'get_root_logger', 'collect_env', 'get_random_string', 'get_thread_id',
    'get_shm_dir', 'GradCAM', 'register_module_hooks',
    'setup_multi_processes', 'build_ddp', 'build_dp', 'default_device',
    'get_logger', 'print_log', '_AdaptiveAvgPoolNd', '_AdaptiveMaxPoolNd', '_AvgPoolNd', '_BatchNorm',
    '_ConvNd', '_ConvTransposeMixin', '_InstanceNorm', '_MaxPoolNd'
]
