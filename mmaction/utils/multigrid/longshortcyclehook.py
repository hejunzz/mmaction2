# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
# from mmcv.runner import Hook
from mmengine.hooks import Hook
# from mmcv.runner.hooks.lr_updater import LrUpdaterHook, StepLrUpdaterHook
from torch.nn.modules.utils import _ntuple

# from mmaction.core.lr import RelativeStepLrUpdaterHook
from mmaction.utils import get_root_logger
from mmaction.misc import is_list_of

# from mmengine.hooks import HOOKS
from ...registry import Registry

HOOKS = Registry('hook')

class LrUpdaterHook(Hook):
    """LR Scheduler in MMCV.

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_by_epoch=False):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, runner):
        if isinstance(runner.optimizer, dict):
            lr_groups = {}
            for k in runner.optimizer.keys():
                _lr_group = [
                    self.get_lr(runner, _base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [
                    group['initial_lr'] for group in optim.param_groups
                ]
                self.base_lr.update({k: _base_lr})
        else:
            for group in runner.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [
                group['initial_lr'] for group in runner.optimizer.param_groups
            ]

    def before_train_epoch(self, runner):
        if self.warmup_iters is None:
            epoch_len = len(runner.data_loader)
            self.warmup_iters = self.warmup_epochs * epoch_len

        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner, batch_idx):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


@HOOKS.register_module()
class StepLrUpdaterHook(LrUpdaterHook):
    """Step LR scheduler with min_lr clipping.

    Args:
        step (int | list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (float, optional): Decay LR ratio. Default: 0.1.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
    """

    def __init__(self, step, gamma=0.1, min_lr=None, **kwargs):
        if isinstance(step, list):
            assert is_list_of(step, int)
            assert all([s > 0 for s in step])
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        super(StepLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr


def modify_subbn3d_num_splits(logger, module, num_splits):
    """Recursively modify the number of splits of subbn3ds in module.
    Inheritates the running_mean and running_var from last subbn.bn.

    Args:
        logger (:obj:`logging.Logger`): The logger to log information.
        module (nn.Module): The module to be modified.
        num_splits (int): The targeted number of splits.
    Returns:
        int: The number of subbn3d modules modified.
    """
    count = 0
    for child in module.children():
        from mmaction.models import SubBatchNorm3D
        if isinstance(child, SubBatchNorm3D):
            new_split_bn = nn.BatchNorm3d(
                child.num_features * num_splits, affine=False).cuda()
            new_state_dict = new_split_bn.state_dict()

            for param_name, param in child.bn.state_dict().items():
                origin_param_shape = param.size()
                new_param_shape = new_state_dict[param_name].size()
                if len(origin_param_shape) == 1 and len(
                        new_param_shape
                ) == 1 and new_param_shape[0] >= origin_param_shape[
                        0] and new_param_shape[0] % origin_param_shape[0] == 0:
                    # weight bias running_var running_mean
                    new_state_dict[param_name] = torch.cat(
                        [param] *
                        (new_param_shape[0] // origin_param_shape[0]))
                else:
                    logger.info(f'skip  {param_name}')

            child.num_splits = num_splits
            new_split_bn.load_state_dict(new_state_dict)
            child.split_bn = new_split_bn
            count += 1
        else:
            count += modify_subbn3d_num_splits(logger, child, num_splits)
    return count

@HOOKS.register_module()
# Register it here
class RelativeStepLrUpdaterHook(LrUpdaterHook):
    # You should inheritate it from mmcv.LrUpdaterHook
    def __init__(self, steps, lrs, **kwargs):
        super().__init__(**kwargs)
        assert len(steps) == (len(lrs))
        self.steps = steps
        self.lrs = lrs

    def get_lr(self, runner, base_lr):
        # Only this function is required to override
        # This function is called before each training epoch, return the specific learning rate here.
        progress = runner.epoch if self.by_epoch else runner.iter
        for i in range(len(self.steps)):
            if progress < self.steps[i]:
                return self.lrs[i]


class LongShortCycleHook(Hook):
    """A multigrid method for efficiently training video models.

    This hook defines multigrid training schedule and update cfg
        accordingly, which is proposed in `A Multigrid Method for Efficiently
        Training Video Models <https://arxiv.org/abs/1912.00998>`_.

    Args:
        cfg (:obj:`mmcv.ConfigDictg`): The whole config for the experiment.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.multi_grid_cfg = cfg.get('multigrid', None)
        self.data_cfg = cfg.get('data', None)
        assert (self.multi_grid_cfg is not None and self.data_cfg is not None)
        self.logger = get_root_logger()
        self.logger.info(self.multi_grid_cfg)

    def before_run(self, runner):
        """Called before running, change the StepLrUpdaterHook to
        RelativeStepLrHook."""
        self._init_schedule(runner, self.multi_grid_cfg, self.data_cfg)
        steps = []
        steps = [s[-1] for s in self.schedule]
        steps.insert(-1, (steps[-2] + steps[-1]) // 2)  # add finetune stage
        for index, hook in enumerate(runner.hooks):
            if isinstance(hook, StepLrUpdaterHook):
                base_lr = hook.base_lr[0]
                gamma = hook.gamma
                lrs = [base_lr * gamma**s[0] * s[1][0] for s in self.schedule]
                lrs = lrs[:-1] + [lrs[-2], lrs[-1] * gamma
                                  ]  # finetune-stage lrs
                new_hook = RelativeStepLrUpdaterHook(runner, steps, lrs)
                runner.hooks[index] = new_hook

    def before_train_epoch(self, runner):
        """Before training epoch, update the runner based on long-cycle
        schedule."""
        self._update_long_cycle(runner)

    def _update_long_cycle(self, runner):
        """Before every epoch, check if long cycle shape should change. If it
        should, change the pipelines accordingly.

        change dataloader and model's subbn3d(split_bn)
        """
        base_b, base_t, base_s = self._get_schedule(runner.epoch)

        # rebuild dataset
        from mmaction.datasets import build_dataset
        resize_list = []
        for trans in self.cfg.data.train.pipeline:
            if trans['type'] == 'SampleFrames':
                curr_t = trans['clip_len']
                trans['clip_len'] = base_t
                trans['frame_interval'] = (curr_t *
                                           trans['frame_interval']) / base_t
            elif trans['type'] == 'Resize':
                resize_list.append(trans)
        resize_list[-1]['scale'] = _ntuple(2)(base_s)

        ds = build_dataset(self.cfg.data.train)

        from mmaction.datasets import build_dataloader

        dataloader = build_dataloader(
            ds,
            self.data_cfg.videos_per_gpu * base_b,
            self.data_cfg.workers_per_gpu,
            dist=True,
            num_gpus=len(self.cfg.gpu_ids),
            drop_last=True,
            seed=self.cfg.get('seed', None),
        )
        runner.data_loader = dataloader
        self.logger.info('Rebuild runner.data_loader')

        # the self._max_epochs is changed, therefore update here
        runner._max_iters = runner._max_epochs * len(runner.data_loader)

        # rebuild all the sub_batch_bn layers
        num_modifies = modify_subbn3d_num_splits(self.logger, runner.model,
                                                 base_b)
        self.logger.info(f'{num_modifies} subbns modified to {base_b}.')

    def _get_long_cycle_schedule(self, runner, cfg):
        # `schedule` is a list of [step_index, base_shape, epochs]
        schedule = []
        avg_bs = []
        all_shapes = []
        self.default_size = self.default_t * self.default_s**2
        for t_factor, s_factor in cfg.long_cycle_factors:
            base_t = int(round(self.default_t * t_factor))
            base_s = int(round(self.default_s * s_factor))
            if cfg.short_cycle:
                shapes = [[
                    base_t,
                    int(round(self.default_s * cfg.short_cycle_factors[0]))
                ],
                          [
                              base_t,
                              int(
                                  round(self.default_s *
                                        cfg.short_cycle_factors[1]))
                          ], [base_t, base_s]]
            else:
                shapes = [[base_t, base_s]]
            # calculate the batchsize, shape = [batchsize, #frames, scale]
            shapes = [[
                int(round(self.default_size / (s[0] * s[1]**2))), s[0], s[1]
            ] for s in shapes]
            avg_bs.append(np.mean([s[0] for s in shapes]))
            all_shapes.append(shapes)

        for hook in runner.hooks:
            if isinstance(hook, LrUpdaterHook):
                if isinstance(hook, StepLrUpdaterHook):
                    steps = hook.step if isinstance(hook.step,
                                                    list) else [hook.step]
                    steps = [0] + steps
                    break
                else:
                    raise NotImplementedError(
                        'Only step scheduler supports multi grid now')
            else:
                pass
        total_iters = 0
        default_iters = steps[-1]
        for step_index in range(len(steps) - 1):
            # except the final step
            step_epochs = steps[step_index + 1] - steps[step_index]
            # number of epochs for this step
            for long_cycle_index, shapes in enumerate(all_shapes):
                cur_epochs = (
                    step_epochs * avg_bs[long_cycle_index] / sum(avg_bs))
                cur_iters = cur_epochs / avg_bs[long_cycle_index]
                total_iters += cur_iters
                schedule.append((step_index, shapes[-1], cur_epochs))
        iter_saving = default_iters / total_iters
        final_step_epochs = runner.max_epochs - steps[-1]
        # the fine-tuning phase to have the same amount of iteration
        # saving as the rest of the training
        ft_epochs = final_step_epochs / iter_saving * avg_bs[-1]
        # in `schedule` we ignore the shape of ShortCycle
        schedule.append((step_index + 1, all_shapes[-1][-1], ft_epochs))

        x = (
            runner.max_epochs * cfg.epoch_factor / sum(s[-1]
                                                       for s in schedule))
        runner._max_epochs = int(runner._max_epochs * cfg.epoch_factor)
        final_schedule = []
        total_epochs = 0
        for s in schedule:
            # extend the epochs by `factor`
            epochs = s[2] * x
            total_epochs += epochs
            final_schedule.append((s[0], s[1], int(round(total_epochs))))
        self.logger.info(final_schedule)
        return final_schedule

    def _print_schedule(self, schedule):
        """logging the schedule."""
        self.logger.info('\tLongCycleId\tBase shape\tEpochs\t')
        for s in schedule:
            self.logger.info(f'\t{s[0]}\t{s[1]}\t{s[2]}\t')

    def _get_schedule(self, epoch):
        """Returning the corresponding shape."""
        for s in self.schedule:
            if epoch < s[-1]:
                return s[1]
        return self.schedule[-1][1]

    def _init_schedule(self, runner, multi_grid_cfg, data_cfg):
        """Initialize the multigrid shcedule.

        Args:
            runner (:obj: `mmcv.Runner`): The runner within which to train.
            multi_grid_cfg (:obj: `mmcv.ConfigDict`): The multigrid config.
            data_cfg (:obj: `mmcv.ConfigDict`): The data config.
        """
        self.default_bs = data_cfg.videos_per_gpu
        data_cfg = data_cfg.get('train', None)
        final_resize_cfg = [
            aug for aug in data_cfg.pipeline if aug.type == 'Resize'
        ][-1]
        if isinstance(final_resize_cfg.scale, tuple):
            # Assume square image
            if max(final_resize_cfg.scale) == min(final_resize_cfg.scale):
                self.default_s = max(final_resize_cfg.scale)
            else:
                raise NotImplementedError('non-square scale not considered.')
        sample_frame_cfg = [
            aug for aug in data_cfg.pipeline if aug.type == 'SampleFrames'
        ][0]
        self.default_t = sample_frame_cfg.clip_len

        if multi_grid_cfg.long_cycle:
            self.schedule = self._get_long_cycle_schedule(
                runner, multi_grid_cfg)
        else:
            raise ValueError('There should be at least long cycle.')
