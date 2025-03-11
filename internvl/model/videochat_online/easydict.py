class EasyDict(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in ("update", "pop"):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasyDict, self).pop(k, d)

video_tower_config = {
    'vision_encoder': {
        'name': "vit_l14",
        'img_size': 224,
        'patch_size': 16,
        'd_model': 1024,
        'encoder_embed_dim': 1024,
        'encoder_depth': 24,
        'encoder_num_heads': 16,
        'drop_path_rate': 0.0,
        'num_frames': 4,
        'tubelet_size': 1,
        'use_checkpoint': False,
        'checkpoint_num': 0,
        'pretrained': None,
        'return_index': -2,
        'vit_add_ln': True
    },
    'use_grad_checkpoint': True,
    'use_lora': False,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'use_flash_attention': True,
    'freeze_vision_tower': True,
    'freeze_vit': False,
    'freeze_qformer': False,
    'vit_blip_model_path': '/mnt/petrelfs/share_data/likunchang/model/videochat2/umt_l16_qformer.pth',
    'low_resource': False,
    'num_query_token': 32,
    'qformer_hidden_dropout_prob': 0.1,
    'qformer_attention_probs_dropout_prob': 0.1,
    'qformer_drop_path_rate': 0.2,
    'extra_num_query_token': 64,
    'max_txt_len': 32
}

cfg = EasyDict(video_tower_config)

#import pdb
#pdb.set_trace()