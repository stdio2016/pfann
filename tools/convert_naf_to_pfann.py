# Copy this program to neural-audio-fp repo from https://github.com/mimbres/neural-audio-fp
from collections import OrderedDict
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import torch
import numpy as np
import tensorflow as tf
import argparse

# from neural-audio-fp repo
from model.utils.config_gpu_memory_lim import allow_gpu_memory_growth
import run
from model.generate import build_fp, load_checkpoint

def convert_conv2d(conv, prefix, out):
    out[prefix + '.weight'] = conv.get_weights()[0].transpose([3, 2, 0, 1])
    out[prefix + '.bias'] = conv.get_weights()[1]

def convert_layernorm(ln, prefix, out):
    out[prefix + '.weight'] = ln.get_weights()[0].transpose([2, 0, 1])
    out[prefix + '.bias'] = ln.get_weights()[1].transpose([2, 0, 1])

def convert_conv_layer(conv, prefix, out):
    convert_conv2d(conv.conv2d_1x3, prefix + '.conv1', out)
    convert_layernorm(conv.BN_1x3, prefix + '.ln1', out)
    convert_conv2d(conv.conv2d_3x1, prefix + '.conv2', out)
    convert_layernorm(conv.BN_3x1, prefix + '.ln2', out)
    return conv.conv2d_1x3.strides, conv.conv2d_3x1.strides

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('checkpoint_name')
    args.add_argument('--checkpoint-index')
    args.add_argument('--config', default='default')
    args.add_argument('pfann')
    args = args.parse_args()
    
    cfg = run.load_config(args.config)
    
    # copied from https://github.com/mimbres/neural-audio-fp/blob/main/model/generate.py
    # load model from checkpoint
    m_pre, m_fp = build_fp(cfg)
    model = tf.train.Checkpoint(model=m_fp)
    checkpoint_root_dir = cfg['DIR']['LOG_ROOT_DIR'] + 'checkpoint/'
    checkpoint_index = load_checkpoint(checkpoint_root_dir, args.checkpoint_name,
                                       args.checkpoint_index, m_fp)
    
    n_frame = int(cfg['MODEL']['DUR'] * cfg['MODEL']['FS'])
    
    # initialize model
    x = np.zeros([1, 1, n_frame])
    y = m_fp(m_pre(x)).numpy()
    
    # convert weight
    out = OrderedDict()
    strides = []
    for lv, conv in enumerate(m_fp.front_conv.layers[:-1]):
        stride = convert_conv_layer(conv, 'f.convs.%d' % lv, out)
        strides.append(stride)
    h = list(out.items())[-1][1].shape[0]
    fc1w = []
    fc1b = []
    fc2w = []
    fc2b = []
    for seq in m_fp.div_enc.split_fc_layers:
        fc1w.append(seq.layers[0].weights[0])
        fc1b.append(seq.layers[0].weights[1])
        u = seq.layers[0].weights[1].shape[0]
        fc2w.append(seq.layers[1].weights[0])
        fc2b.append(seq.layers[1].weights[1])
    out['g.linear1.weight'] = np.expand_dims(np.concatenate(fc1w, axis=1).T, 2)
    out['g.linear1.bias'] = np.concatenate(fc1b)
    out['g.linear2.weight'] = np.expand_dims(np.concatenate(fc2w, axis=1).T, 2)
    out['g.linear2.bias'] = np.concatenate(fc2b)
    out = {x:torch.from_numpy(out[x]) for x in out}
    
    # save weight
    os.makedirs(args.pfann, exist_ok=True)
    torch.save(out, os.path.join(args.pfann, 'model.pt'))
    params = {
        "model_dir": args.pfann,
        "fftconv_n": 32768,
        "sample_rate": cfg['MODEL']['FS'],
        "stft_n": cfg['MODEL']['STFT_WIN'],
        "stft_hop": cfg['MODEL']['STFT_HOP'],
        "n_mels": cfg['MODEL']['N_MELS'],
        "dynamic_range": 80,
        "f_min": cfg['MODEL']['F_MIN'],
        "f_max": cfg['MODEL']['F_MAX'],
        "segment_size": cfg['MODEL']['DUR'],
        "hop_size": cfg['MODEL']['HOP'],
        "naf_mode": True,
        "mel_log": "log10",
        "spec_norm": "max",
        "model": {
            "d": cfg['MODEL']['EMB_SZ'],
            "h": h,
            "u": u,
            "fuller": True,
            "conv_activation": "ELU",
            "relu_after_bn": False,
            "strides": strides,
        },
        "indexer": {
            "index_factory": "IVF200,PQ64x8np",
            "top_k": 100,
        }
    }
    with open(os.path.join(args.pfann, 'configs.json'), 'w') as fout:
        json.dump(params, fout, indent=2)
