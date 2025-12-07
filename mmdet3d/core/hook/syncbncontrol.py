# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel
from torch.nn import SyncBatchNorm

__all__ = ['SyncbnControlHook']


@HOOKS.register_module()
class SyncbnControlHook(Hook):
    """ """

    def __init__(self, syncbn_start_epoch=1):
        super().__init__()
        self.is_syncbn=False
        self.syncbn_start_epoch = syncbn_start_epoch

    def cvt_syncbn(self, runner):
        # If distributed is not initialized (single GPU / non-dist), skip conversion.
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            print('Skip SyncBN conversion: torch.distributed not initialized, keep BatchNorm.')
            return False
        if is_parallel(runner.model.module):
            runner.model.module.module = \
                SyncBatchNorm.convert_sync_batchnorm(runner.model.module.module,
                                                     process_group=None)
        else:
            runner.model.module = \
                SyncBatchNorm.convert_sync_batchnorm(runner.model.module,
                                                     process_group=None)
        return True

    def before_train_epoch(self, runner):
        if runner.epoch>= self.syncbn_start_epoch and not self.is_syncbn:
            print('start use syncbn')
            converted = self.cvt_syncbn(runner)
            if converted:
                print('syncbn enabled')
            self.is_syncbn=True

