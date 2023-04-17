
python tools/test.py configs/mask_rcnn/mask_rcnn_r50_fpn_2x_foots.py work_dirs/mask_rcnn_r50_fpn_2x_foots/epoch_10.pth --eval bbox segm --work-dir results/mAP --show --show-dir results/visualize_images
CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/mask_rcnn/mask_rcnn_r50_fpn_2x_foots.py work_dirs/mask_rcnn_r50_fpn_2x_foots/epoch_10.pth --eval bbox segm --work-dir results/mAP

CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mask_rcnn/mask_rcnn_r50_fpn_2x_foots.py work_dirs/mask_rcnn_r50_fpn_2x_foots_AL_9/epoch_33.pth --show

CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mask2former/mask2former_r50_lsj_8x2_50e_foots.py

CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/mask_rcnn/mask_rcnn_r50_fpn_2x_foots_full_images.py

CUDA_VISIBLE_DEVICES=1 python tools/train_AL.py configs/mask_rcnn/mask_rcnn_r50_fpn_2x_foots.py