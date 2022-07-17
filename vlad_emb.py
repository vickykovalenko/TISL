# NetVLAD
from __future__ import print_function
import random
from os.path import join, isfile

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import netvlad

def get_vlad_emb(img):
    cuda = False
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    default_seed = 123
    random.seed(default_seed)
    np.random.seed(default_seed)
    torch.manual_seed(default_seed)
    if cuda:
        torch.cuda.manual_seed(default_seed)

#     print('===> Loading data')
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])
    input = preprocess(img)
    input = input.unsqueeze(0)
#     print('Input image shape: {}'.format(input.shape))

#     print('===> Building model')

    pretrained = True

    encoder_dim = 512
    encoder = models.vgg16(pretrained=pretrained)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]

    if pretrained:
        # if using pretrained then only train conv5_1, conv5_2, and conv5_3
        for l in layers[:-5]: 
            for p in l.parameters():
                p.requires_grad = False


    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)

    default_num_clusters = 64
    defauly_vladv2 = False
#     print('NetVLAD setting:\nnum_clusters: {} dim: {} vladv2: {}'.format(default_num_clusters, encoder_dim, defauly_vladv2))
    net_vlad = netvlad.NetVLAD(num_clusters=default_num_clusters, dim=encoder_dim, vladv2=defauly_vladv2)
    model.add_module('pool', net_vlad)

    checkpoint_path = 'vgg16_netvlad_checkpoint'
    resume_ckpt = join(checkpoint_path, 'checkpoints', 'checkpoint_512_64.pth.tar')

    if isfile(resume_ckpt):
#         print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['best_score']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
#         print("=> loaded checkpoint '{}' (epoch {})"
#                 .format(resume_ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_ckpt))

#     print('===> Running evaluation step')
    model.eval()
    with torch.no_grad():
        input = input.to(device)
        image_encoding = model.encoder(input)
        vlad_encoding = model.pool(image_encoding) 
#     print('VLAD encoding: {}'.format(vlad_encoding))
#     print('VLAD encoding shape: {}'.format(vlad_encoding.shape))
    return vlad_encoding