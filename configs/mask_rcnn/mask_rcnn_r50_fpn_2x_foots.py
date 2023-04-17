# The new config inherits a base config to highlight the necessary modification
_base_ = 'mask_rcnn_r50_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=32),
        mask_head=dict(num_classes=32)))

# Modify dataset related settings
dataset_type = 'COCODataset'
# classes = ('balloon',)
classes = ("Met3", "MedPhal3", "Accessory", "Fibula", "Talus", "Calcaneus", "Cuboid", "Cuneiform Medial", "Medial Sesamoid", "ProxPhal2", "Navicular", "Tibia", "Cuneiform Intermediate", "ProxPhal1", "Met1", "ProxPhal3", "Cuneiform Lateral", "ProxPhal4", "DisPhal5", "DisPhal2", "Met2", "DisPhal3", "DisPhal4", "Met4", "ProxPhal5", "MedPhal2", "Lateral Sesamoid", "Met5", "MedPhal5", "DisPhal1", "MedPhal4", "Auxiliary Sesamoid")
root = '/home/hungld11/Intern/khanghn1/foot_dataset/'
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    train=dict(
        # img_prefix=root + 'images/train/',
        img_prefix=root + 'AL/images/',
        classes=classes,
        # ann_file=root + 'foot_dataset_train.json'),
        ann_file=root + 'AL/foot_dataset_AL.json'),
    val=dict(
        img_prefix=root + 'images/val/',
        classes=classes,
        ann_file=root + 'foot_dataset_val.json'),
    test=dict(
        img_prefix=root + 'images/val/',
        classes=classes,
        ann_file=root + 'foot_dataset_val.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
# load_from = 'work_dirs/mask_rcnn_r50_fpn_2x_foots/epoch_24.pth'
# resume_from = 'checkpoints/mask_rcnn_r50_fpn_mocov2-pretrain_ms-2x_coco_20210605_163717-d95df20a.pth'