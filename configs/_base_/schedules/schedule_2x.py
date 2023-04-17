# optimizer
# optimizer = dict(type='SGD', lr=2e-5, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[20, 45, 75, 100])
    # step=[15, 30, 50])
    # step=[24, 50, 80, 145])
# lr_config = dict(
#     policy='OneCycleLr',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.01,
#     min_lr=1e-7,
#     min_lr_ratio=1e-5)

runner = dict(type='EpochBasedRunner', max_epochs=100)
