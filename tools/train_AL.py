# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import shutil

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)


from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
root = '/home/hungld11/Intern/khanghn1/foot_dataset/'

def calculate_uncertainty(model, data_loader, cfg, distributed=False):
    print('===>>> Calculating instance uncertainty: ...')
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    uncertainty = np.zeros(len(dataset))
    ori_filenames = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            res_boxes, res_masks = model(return_loss=False, rescale=True, **data)[0]

        uncertainty_single = []
        for box in res_boxes:
            if box.size == 0: continue
            elif box.size == 1:
                uncertainty.append(box[-1])
            if box.size > 0:
                for b in box:
                    uncertainty_single.append(b[-1])
        uncertainty[i] = np.mean(uncertainty_single)
        ori_filenames.append(data['img_metas'][0].data[0][0]['ori_filename'])
    idx = np.argsort(uncertainty)
    print('len = ', len(ori_filenames), uncertainty.size)
    for sample in np.array(ori_filenames)[idx[:200]]:
        shutil.copy2(sample, root + f'AL/images/')
        shutil.copy2(sample.replace('images', 'masks'), root + f'AL/masks/')
    print('len training data after adding 200 labelled samples: ', len(os.listdir(root + 'AL/images/')), len(os.listdir(root + 'AL/masks/')))
    convert_to_coco_format()

def convert_to_coco_format(batch=None):
    print('===>>> Converting to coco format:')
    num_classes = 32
    categories = {"__background__": 0, "Met3": 1, "MedPhal3": 2, "Accessory": 3, "Fibula": 4, "Talus": 5, "Calcaneus": 6, "Cuboid": 7, "Cuneiform Medial": 8, "Medial Sesamoid": 9, "ProxPhal2": 10, "Navicular": 11, "Tibia": 12, "Cuneiform Intermediate": 13, "ProxPhal1": 14, "Met1": 15, "ProxPhal3": 16, "Cuneiform Lateral": 17, "ProxPhal4": 18, "DisPhal5": 19, "DisPhal2": 20, "Met2": 21, "DisPhal3": 22, "DisPhal4": 23, "Met4": 24, "ProxPhal5": 25, "MedPhal2": 26, "Lateral Sesamoid": 27, "Met5": 28, "MedPhal5": 29, "DisPhal1": 30, "MedPhal4": 31, "Auxiliary Sesamoid": 32}
    if batch is not None:
        ROOT = f'/home/hungld11/Intern/khanghn1/foot_dataset/AL/unlabel_batch_{batch}/'
    else:
        ROOT = '/home/hungld11/Intern/khanghn1/foot_dataset/AL/'
    json_file_output = ROOT + "foot_dataset_AL.json"
    
    # load all image files, sorting them` to ensure that they are aligned
    imgs_path = np.array(sorted(os.path.join(ROOT, "images", i) for i in os.listdir(os.path.join(ROOT, "images"))))
    #load all mask files, sorting them to ensure that they are aligned
    masks_path = np.array(sorted(os.path.join(ROOT, "masks", i) for i in os.listdir(os.path.join(ROOT, "masks"))))

    data = dict(
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    cnt = 0
    for idx, (img_pth, mask_pth) in tqdm(enumerate(zip(imgs_path, masks_path))):
        image = cv2.imread(img_pth)
        if image is None:
            print("Error image: ", img_pth)
        mask = np.array(Image.open(mask_pth))
        cnt += mask.max() > 32
        if mask.max() > 32:
            print("Error mask: ", mask_pth)
            print(mask.max(), mask.min())
        # mask[mask > 32] = 0
        # cv2.imwrite(mask_pth, mask)
        height, width = mask.shape
        data['images'].append(dict(
            id=idx,
            height=height,
            width=width,
            file_name=img_pth,
        ))
        
        obj_ids = np.unique(mask)[1:]

        masks = mask == obj_ids[:, None, None]
        # print('1 = ', masks.shape)

        for i, mask in enumerate(masks):    
            if mask.sum() == 0: continue
            # print('2 = ', mask.shape)

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.reshape(-1, 2)
                # if contour.shape[0] == 1: contour = np.concatenate([contour, contour], axis=0)
                if contour.shape[0] <= 2: continue
                x1, y1 = contour.min(0)
                x2, y2 = contour.max(0)
                bbox = np.array([x1, y1, x2 - x1, y2 - y1])
                area = np.sum(mask[y1:y2+1, x1:x2+1])
                data['annotations'].append(dict(
                    id = len(data['annotations']),
                    image_id=idx,
                    category_id=obj_ids[i],
                    segmentation=[contour.reshape(-1).tolist()],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                ))
        # break
    print('number images > 32: ', cnt)

    for class_name, class_id in categories.items():
        data["categories"].append(
            dict(
                id=class_id,
                name=class_name,
            )
        )
    # with open(json_file_output, 'w') as f:
    #     json.dump(data, f)
    mmcv.dump(data, json_file_output)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    #Check path exsist or nor   
    if not osp.exists(root + 'AL/images/'):
        os.mkdir(root + 'AL/images/')
        os.mkdir(root + 'AL/masks/')
        
    CYCLE = 10
    for cycle in range(9, CYCLE):
        print('CYCLE: ', cycle)

        # best_acc = 0

        # # open 356-batch (sorted low->high)
        # with open(f'/home/hungld11/Intern/khanghn1/PT4AL/loss/batch_{cycle}.txt', 'r') as f:
        #     samples = np.array(f.readlines())

        # if cycle == 0:
        #     sample100 = samples[[j for j in range(100)]]
        #     for sample in samples:
        #         shutil.copy2(root + 'images/train/' + sample[:-1], root + f'AL/images/')
        #         shutil.copy2(root + 'masks/train/' + sample[:-1], root + f'AL/masks/')
        #     convert_to_coco_format()
        # else:
        #     if not osp.exists(root + f'AL/unlabel_batch_{cycle}/images/'):
        #         os.mkdir(root + f'AL/unlabel_batch_{cycle}')
        #         os.mkdir(root + f'AL/unlabel_batch_{cycle}/images/')
        #         os.mkdir(root + f'AL/unlabel_batch_{cycle}/masks/')
        #     for sample in samples:
        #         # shutil.copy2(root + 'images/train/' + sample[:-1], root + f'AL/unlabel_batch_{cycle}/images/')
        #         # shutil.copy2(root + 'masks/train/' + sample[:-1], root + f'AL/unlabel_batch_{cycle}/masks/')
        #         mask = cv2.imread(root + 'masks/train/' + sample[:-1])
        #         if mask.max() > 32:
        #             print('Errorrrrrrrrr: ', root + 'masks/train/' + sample[:-1])
        #             print(mask.max(), mask.min())
        #         mask = cv2.imread(root + f'AL/unlabel_batch_{cycle}/masks/' + sample[:-1])
        #         if mask.max() > 32:
        #             print('Errorrrrrrrrr: ', root + f'AL/unlabel_batch_{cycle}/masks/' + sample[:-1])
        #             print(mask.max(), mask.min())

        #         image = cv2.imread(root + f'AL/unlabel_batch_{cycle}/images/' + sample[:-1])
        #         if image is None:
        #             print('Errorrrrrrrrr: ', root + f'AL/unlabel_batch_{cycle}/images/' + sample[:-1])
        #     print(f'len of unlabel batch {cycle}:', len(os.listdir(root + f'AL/unlabel_batch_{cycle}/images/')), len(os.listdir(root + f'AL/unlabel_batch_{cycle}/masks/')))
        #     convert_to_coco_format(cycle)
            
        #     agrs = parse_args()
        #     cfg = Config.fromfile(args.config)

        #     # replace the ${key} with the value of cfg.key
        #     cfg = replace_cfg_vals(cfg)

        #     # update data root according to MMDET_DATASETS
        #     update_data_root(cfg)

        #     if args.cfg_options is not None:
        #         cfg.merge_from_dict(args.cfg_options)

        #     cfg = compat_cfg(cfg)

        #     # set multi-process settings
        #     setup_multi_processes(cfg)

        #     # set cudnn_benchmark
        #     if cfg.get('cudnn_benchmark', False):
        #         torch.backends.cudnn.benchmark = True
            
        #     if cfg.model.get('neck'):
        #         if isinstance(cfg.model.neck, list):
        #             for neck_cfg in cfg.model.neck:
        #                 if neck_cfg.get('rfp_backbone'):
        #                     if neck_cfg.rfp_backbone.get('pretrained'):
        #                         neck_cfg.rfp_backbone.pretrained = None
        #         elif cfg.model.neck.get('rfp_backbone'):
        #             if cfg.model.neck.rfp_backbone.get('pretrained'):
        #                 cfg.model.neck.rfp_backbone.pretrained = None

        #     if args.gpu_ids is not None:
        #         cfg.gpu_ids = args.gpu_ids[0:1]
        #         warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
        #                     'Because we only support single GPU mode in '
        #                     'non-distributed testing. Use the first GPU '
        #                     'in `gpu_ids` now.')
        #     else:
        #         cfg.gpu_ids = [args.gpu_id]
                
        #     cfg.device = get_device() 
            
        #     if isinstance(cfg.data.test, dict):
        #         cfg.data.test.test_mode = True
        #         if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
        #             # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        #             cfg.data.test.pipeline = replace_ImageToTensor(
        #                 cfg.data.test.pipeline)
        #     elif isinstance(cfg.data.test, list):
        #         for ds_cfg in cfg.data.test:
        #             ds_cfg.test_mode = True
        #         if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
        #             for ds_cfg in cfg.data.test:
        #                 ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
            
        #     unlabel_dataset_cfg = cfg.data.test.copy()
        #     unlabel_dataset_cfg.img_prefix = root + f'AL/unlabel_batch_{cycle}/images/'
        #     unlabel_dataset_cfg.ann_file = root + f'AL/unlabel_batch_{cycle}/foot_dataset_AL.json'
        #     unlabel_loader_cfg = dict(
        #         samples_per_gpu=1,
        #         # workers_per_gpu=2,
        #         dist=False,
        #         # shuffe=False,
        #         **cfg.data.get('test_dataloader', {})
        #     )
        #     unlabel_dataset = build_dataset(unlabel_dataset_cfg)
        #     print(f'len unlabel_dataset {cycle}: ', len(unlabel_dataset))
        #     unlabel_dataloader = build_dataloader(unlabel_dataset, **unlabel_loader_cfg)
        #     model_AL = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        #     ckpt = 'work_dirs/mask_rcnn_r50_fpn_2x_foots_AL_6/epoch_32.pth'
        #     checkpoint = load_checkpoint(model_AL, ckpt, map_location='cpu')
        #     model_AL = build_dp(model_AL, cfg.device, device_ids=cfg.gpu_ids)
        #     calculate_uncertainty(model_AL, unlabel_dataloader, cfg)

        # if cycle < 8: continue
        cfg = Config.fromfile(args.config)

        # replace the ${key} with the value of cfg.key
        cfg = replace_cfg_vals(cfg)

        # update data root according to MMDET_DATASETS
        update_data_root(cfg)

        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        if args.auto_scale_lr:
            if 'auto_scale_lr' in cfg and \
                    'enable' in cfg.auto_scale_lr and \
                    'base_batch_size' in cfg.auto_scale_lr:
                cfg.auto_scale_lr.enable = True
            else:
                warnings.warn('Can not find "auto_scale_lr" or '
                            '"auto_scale_lr.enable" or '
                            '"auto_scale_lr.base_batch_size" in your'
                            ' configuration file. Please update all the '
                            'configuration files to mmdet >= 2.24.1.')

        # set multi-process settings
        setup_multi_processes(cfg)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.work_dir = f'work_dirs/mask_rcnn_r50_fpn_2x_foots_AL_{cycle}/'

        if args.gpus is None and args.gpu_ids is None:
            cfg.gpu_ids = [args.gpu_id]

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
        meta['config'] = cfg.pretty_text
        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        cfg.device = get_device()
        # set random seeds
        seed = init_random_seed(args.seed, device=cfg.device)
        seed = seed + dist.get_rank() if args.diff_seed else seed
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(seed, deterministic=args.deterministic)
        cfg.seed = seed
        meta['seed'] = seed
        meta['exp_name'] = osp.basename(args.config)

        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model.init_weights()

        # init rfnext if 'RFSearchHook' is defined in cfg
        rfnext_init_model(model, cfg=cfg)

        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            assert 'val' in [mode for (mode, _) in cfg.workflow]
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.get(
                'pipeline', cfg.data.train.dataset.get('pipeline'))
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        # cfg.resume_from = f'work_dirs/mask_rcnn_r50_fpn_2x_foots_AL_9/epoch_33.pth'
        cfg.resume_from = 'checkpoints/mask_rcnn_r50_fpn_mocov2-pretrain_ms-2x_coco_20210605_163717-d95df20a.pth'
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)
if __name__ == '__main__':
    main()
