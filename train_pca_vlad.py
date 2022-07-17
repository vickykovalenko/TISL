from __future__ import print_function
import os
import itertools
from zipfile import ZipFile
from pathlib import Path
import random

import random
from os.path import join, isfile

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import numpy as np
import netvlad
import os

import pickle as pk

from PIL import Image


def VLAD_for_single_image(img):
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

    print('===> Loading data')
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])
    input = preprocess(img)
    input = input.unsqueeze(0)
    print('Input image shape: {}'.format(input.shape))

    print('===> Building model')

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
    print('NetVLAD setting:\nnum_clusters: {} dim: {} vladv2: {}'.format(default_num_clusters, encoder_dim, defauly_vladv2))
    net_vlad = netvlad.NetVLAD(num_clusters=default_num_clusters, dim=encoder_dim, vladv2=defauly_vladv2)
    model.add_module('pool', net_vlad)

    checkpoint_path = 'vgg16_netvlad_checkpoint'
    resume_ckpt = join(checkpoint_path, 'checkpoints', 'checkpoint.pth.tar')

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['best_score']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_ckpt))

    print('===> Running evaluation step')
    model.eval()
    with torch.no_grad():
        input = input.to(device)
        image_encoding = model.encoder(input)
        vlad_encoding = model.pool(image_encoding) 
    print('VLAD encoding: {}'.format(vlad_encoding))
    print('VLAD encoding shape: {}'.format(vlad_encoding.shape))
    return vlad_encoding


def get_subdirectories_names_list(path_to_directory):
    subdirectories_names_list = []
    for root, subdirectories, files in os.walk(path_to_directory):
        for image_subdirectory in subdirectories:
            subdirectories_names_list.append(image_subdirectory)
    return subdirectories_names_list

def get_corresponding_position_file_path_to_a_frame_path(path_to_random_frame):
    frame_name_with_extension = os.path.basename(path_to_random_frame)
    frame_name_without_extension = os.path.splitext(frame_name_with_extension)[0]
    frame_number = os.path.splitext(frame_name_without_extension)[0]
    dir_name_to_frame = os.path.dirname(path_to_random_frame)
    position_file_path = dir_name_to_frame + "/" + frame_number + ".pose.txt"
    return position_file_path

def get_pathes_to_random_frames(image_directory):
    path_to_frames_list = []
    for image_root, image_subdirectories, image_files in os.walk(image_directory):
        image_files.sort()
        for image_file in image_files:
            if 'jpg' in image_file:
                path_to_the_frame = image_directory + "/" + image_file
                path_to_frames_list.append(path_to_the_frame)
    number_of_random_frames = 50
    if len(path_to_frames_list)>50:
        pathes_to_random_frames = random.sample(path_to_frames_list, number_of_random_frames)
        return pathes_to_random_frames
    else:
        return path_to_frames_list

def write_into_txt_file_frame_position_and_frame_embedding(path_to_directory, path_to_random_frame, pca_reload):
    path_to_frame_position = get_corresponding_position_file_path_to_a_frame_path(path_to_random_frame)
    image_subdirectory = path_to_random_frame.split("/")[-2]
    embedding_path = path_to_directory + "/" + f"{image_subdirectory}.txt"
    embedding_file = open(embedding_path,'a')
    position_matrix = np.loadtxt(path_to_frame_position)
    position = torch.from_numpy(position_matrix[0:3, -1])
    frame_position_numpy = position.cpu().detach().numpy()
    np.savetxt(embedding_file, frame_position_numpy, fmt='%1.8f', newline = " ")
    print("Saved frame_position_numpy to embedding_file: ", embedding_file)
    frame = Image.open(path_to_random_frame)
    frame_vlad_embedding = VLAD_for_single_image(frame)
    frame_vlad_embedding_pca = pca_reload.transform(frame_vlad_embedding)
    np.savetxt(embedding_file, frame_vlad_embedding_pca, fmt='%1.8f', delimiter=' ')
    print("Saved frame_vlad_embedding_pca to embedding_file: ", embedding_file)
    embedding_file.close()

def write_into_txt_file_all_files(path_to_directory):
    pca_reload = pk.load(open("pca2.pkl",'rb'))
    rscan_subdirectories_list = get_subdirectories_names_list(path_to_directory)
    #rscan_subdirectories_list_train = rscan_subdirectories_list[:1]
    count = 0
    for rscan_subdirectory in rscan_subdirectories_list:
        print(f'processing {count}/{len(rscan_subdirectories_list)}')
        image_subdirectory = path_to_directory + "/" + rscan_subdirectory
        pathes_to_random_frames = get_pathes_to_random_frames(image_subdirectory)
        pathes_to_random_frames.sort()
        for path_to_random_frame in pathes_to_random_frames:
            print('path_to_random_frame: ', path_to_random_frame)
            write_into_txt_file_frame_position_and_frame_embedding(path_to_directory, path_to_random_frame, pca_reload)
    return pathes_to_random_frames

def main():
    rscan_path = '/home/igor/Desktop/Victoria/tisl/tisl_localization_22s_copy-main/3rscan'
    pathes_to_random_frames = write_into_txt_file_all_files(rscan_path)
    
main()