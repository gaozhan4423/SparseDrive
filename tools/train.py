# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import sys
import os

print(sys.executable, os.path.abspath(__file__))
# import init_paths # for conda pkgs submitting method
import argparse
import copy
import mmcv
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from torch import distributed as dist
from datetime import timedelta

import cv2

cv2.setNumThreads(8)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--resume-from", help="the checkpoint file to resume from"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use "
        "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument("--gpus-per-machine", type=int, default=8)
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi", "mpi_nccl"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both specified, "
            "--options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # 如果cfg_options被修改了，那么会将修改值合并到cfg中
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    # 检查cfg对象中是否有plugin属性，然后再检查该属性的值是否为True
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir # "projects/mmdet3d_plugin/"
                _module_dir = os.path.dirname(plugin_dir) # projects/mmdet3d_plugin
                _module_dir = _module_dir.split("/") # ["projects", "mmdet3d_plugin"]
                _module_path = _module_dir[0]  # 结果是 "projects"

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                # 结果是_module_path = "projects.mmdet3d_plugin"
                print(_module_path)
                plg_lib = importlib.import_module(_module_path) # plugin library导入
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            from projects.mmdet3d_plugin.apis.train import custom_train_model

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.resume_from is not None:
        # 如果恢复路径不为空的话，则将cfg.resume_from设置为args.resume_from
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        # 如果指定了使用的gpu_ids（如[0,1,3], 则将cfg.gpu_ids设置为args.gpu_ids
        cfg.gpu_ids = args.gpu_ids
    else:
        # 如果没有指定gpu_ids, 则执行以下三元判断式，先判断args.gpus是否为None
        # 如果args.gpus不为None，则将cfg.gpu_ids设置为range(args.gpus),否则默认为range(1)即[0]
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # 检查是否启用了自动缩放学习率的选项
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        # 根据论文中的线性缩放原则，当批量增大时，学习率也要增大
        cfg.optimizer["lr"] = cfg.optimizer["lr"] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        # 不使用分布式训练，在单个进程中运行
        distributed = False
    elif args.launcher == "mpi_nccl":
        distributed = True

        import mpi4py.MPI as MPI

        comm = MPI.COMM_WORLD
        mpi_local_rank = comm.Get_rank()
        mpi_world_size = comm.Get_size()
        print(
            "MPI local_rank=%d, world_size=%d"
            % (mpi_local_rank, mpi_world_size)
        )

        # num_gpus = torch.cuda.device_count()
        device_ids_on_machines = list(range(args.gpus_per_machine))
        str_ids = list(map(str, device_ids_on_machines))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)
        torch.cuda.set_device(mpi_local_rank % args.gpus_per_machine)

        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=mpi_world_size,
            rank=mpi_local_rank,
            timeout=timedelta(seconds=3600),
        )

        cfg.gpu_ids = range(mpi_world_size)
        print("cfg.gpu_ids:", cfg.gpu_ids)
    else:
        distributed = True
        # dist_params = dict(backend="nccl")，即使用NVIDIA NCCL作为后端
        init_dist(
            args.launcher, timeout=timedelta(seconds=3600), **cfg.dist_params 
        )
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()# 获得分布式环境中的总进程数（world_size）
        cfg.gpu_ids = range(world_size)

    # 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level
    )

    # 初始化meta字典来记录重要信息，如环境信息和种子，这些将被记录到日志中
    meta = dict()
    # 记录环境信息
    env_info_dict = collect_env()
    # 用列表推导式格式化环境信息得到列表，然后用换行符连接成字符串
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info(
        "Environment info:\n" + dash_line + env_info + "\n" + dash_line
    )
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, "
            f"deterministic: {args.deterministic}"
        )
        # 如果设置了随机种子并且使用确定性选项，则设置随机种子
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed
    # 记录实验名称
    meta["exp_name"] = osp.basename(args.config)

    # 构建检查器
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights() # 初始化模型权重
    logger.info(f"Model:\n{model}")

    cfg.data.train.work_dir = cfg.work_dir
    cfg.data.val.work_dir = cfg.work_dir
    datasets = [build_dataset(cfg.data.train)]

    # 检查工作流配置长度是否为2，如果是的话意味着训练+验证（即有验证集）
    '''
    # 只有训练阶段
    workflow = [('train', 1)]  # 长度为1

    # 训练和验证阶段, 每训练1个epoch就验证1次
    workflow = [('train', 1), ('val', 1)]  # 长度为2
    '''
    if len(cfg.workflow) == 2:
        # 深度复制cfg.data.val配置
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if "dataset" in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        # 设置验证集的test_mode为False, 则不会加载标注数据
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    # 检查是否有检查点配置
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
        )
    # add an attribute for visualization convenience
    # 将数据集的类别名称添加到模型中
    model.CLASSES = datasets[0].CLASSES
    # 检查cfg中是否有plugin属性，有的话则用custom_train_model函数(包含特定插件的自定义训练逻辑)训练模型
    if hasattr(cfg, "plugin"):
        custom_train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta,
        )
    else:
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method(
        "fork"
    )  # use fork workers_per_gpu can be > 1
    main()
