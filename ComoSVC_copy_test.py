import os
import torch
import torch.nn as nn
import yaml
from Vocoder import Vocoder
from como import Como


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

    
def load_model_vocoder(
        model_path,
        device='cpu',
        config_path = None,
        total_steps=1,
        teacher=False
        ):
    if config_path is None:
        config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    else:
        config_file = config_path

    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # load model
    model = ComoSVC(
                args.data.encoder_out_channels, 
                # args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                total_steps,
                teacher      
                )
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'],strict=False)
    model.eval()
    return model, vocoder, args


class ComoSVC(nn.Module):
    def __init__(
            self,
            input_channel,
            # n_spk,
            use_pitch_aug=True,
            out_dims=128, # define in como
            n_layers=20, 
            n_chans=384, 
            n_hidden=100,
            total_steps=1,
            teacher=True
            ):
        super().__init__()

        # self.spk_embed = nn.Linear(input_channel, n_hidden)
        self.spk_proj = nn.Linear(192, n_hidden)

        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        self.teacher=teacher

        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        # self.n_spk = n_spk
        # if n_spk is not None and n_spk > 1:
        #     self.spk_embed = nn.Embedding(n_spk, n_hidden)
        self.n_hidden = n_hidden
        self.decoder = Como(out_dims, n_layers, n_chans, n_hidden, total_steps, teacher) 
        self.input_channel = input_channel

    def forward(self, units, f0, volume, audio_path, aug_shift = None, 
                gt_spec=None, infer=True):
          
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        # print(units.shape)
        x = self.unit_embed(units) + self.f0_embed((1+ f0 / 700).log()) + self.volume_embed(volume)
        # print(x.shape)
        # print(audio_path.shape)
        # tt
        audio_path = audio_path.repeat(1, x.shape[1], 1)
        # audio_path = audio_path.expand(-1, x.shape[1], -1)
        spk_embed = self.spk_proj(audio_path)
        # print()

        x = x + spk_embed #self.spk_proj(audio_path)  


        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5) 
        
        if not infer:
            output  = self.decoder(gt_spec,x,infer=False)       
        else:
            output = self.decoder(gt_spec,x,infer=True)

        return output

