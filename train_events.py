import argparse
import collections

import numpy as np
import time
import math
import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torchvision import transforms
from wandb_logger import WandbLogger

from retinanet import model
from retinanet.dataloader import CSVDataset_event, collater, collater_raw, \
    Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet.dataloader_raw import CSVDataset_event_raw
from torch.utils.data import DataLoader

# from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main(args=None):
    base_dir = '/ws/data/DSEC' #'/home/abhishek/connect'
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_worker', default=4, type=int)
    parser.add_argument('--event_type', default='voxel', help='Event type, voxel grid or raw events')
    parser.add_argument('--event_k', default=1, type=int, help='Extract event k')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', default=f'/ws/external/DSEC_detection_labels/labels_filtered_train.csv',
                        help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default=f'/ws/external/DSEC_detection_labels/labels_filtered_map.csv',
                        help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', default=f'/ws/external/DSEC_detection_labels/labels_filtered_val.csv',
                        help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--csv_test', default=f'/ws/external/DSEC_detection_labels/labels_filtered_test.csv',
                        help='Path to file containing test annotations (optional, see readme)')
    parser.add_argument('--root_img',default=f'{base_dir}/train', help='dir to root rgb images')
    parser.add_argument('--root_event', default=f'{base_dir}/train', help='dir to root event files in dsec directory structure')
    # parser.add_argument('--root_img',default=f'{base_dir}/train/transformed_images',help='dir to root rgb images')
    # parser.add_argument('--root_event', default=f'{base_dir}/DSEC_events_img',help='dir to toot event files in dsec directory structure')
    parser.add_argument('--fusion', help='Type of fusion:1)early_fusion, fpn_fusion, multi-level', type=str, default='fpn_fusion')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=60)
    parser.add_argument('--continue_training', help='load a pretrained file', default=False)
    parser.add_argument('--checkpoint', help='location of pretrained file', default='./csv_dropout_retinanet_63.pt')
    parser.add_argument('--pretrained', action='store_true', help='pretrained model')
    parser.add_argument('--save', type=str, default='debug')
    parser.add_argument('--wandb', action='store_true', default=False, help='log with wandb')


    parser = parser.parse_args(args)

    if parser.wandb:
        wandb_config = dict(project="event_fusion", entity='kaist-url-ai28', name=parser.save)
        wandb_logger = WandbLogger(wandb_config, args)
        wandb_logger.before_run()
    else:
        wandb_logger = None

    save_dir = f'/ws/external/checkpoints/{parser.save}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        if parser.event_type == 'voxel':
            dataset_train = CSVDataset_event(train_file=parser.csv_train,
                                             class_list=parser.csv_classes,
                                             root_event_dir=parser.root_event,
                                             root_img_dir=parser.root_img,
                                             transform=transforms.Compose([Normalizer(), Resizer()]))

            # if parser.csv_val is None:
            #     dataset_val = None
            #     print('No validati`on annotations provided.')
            # else:

            # dataset_val = CSVDataset_event(train_file=parser.csv_val,
            #                                class_list=parser.csv_classes,
            #                                root_event_dir=parser.root_event+'/val',
            #                                root_img_dir=parser.root_img+'/val',
            #                                transform=transforms.Compose([Normalizer(), Resizer()]))
            dataset_test = CSVDataset_event(train_file=parser.csv_test,
                                           class_list=parser.csv_classes,
                                           root_event_dir=parser.root_event,
                                           root_img_dir=parser.root_img,
                                           transform=transforms.Compose([Normalizer(), Resizer()]))

            collater_fn = collater

        elif parser.event_type == 'raw':
            dataset_train = CSVDataset_event_raw(train_file=parser.csv_train,
                                                 class_list=parser.csv_classes,
                                                 root_event_dir=parser.root_event,
                                                 root_img_dir=parser.root_img,
                                                 transform=transforms.Compose([Normalizer(), Resizer()]),
                                                 event_k=parser.event_k)

            dataset_test = CSVDataset_event_raw(train_file=parser.csv_test,
                                                class_list=parser.csv_classes,
                                                root_event_dir=parser.root_event,
                                                root_img_dir=parser.root_img,
                                                transform=transforms.Compose([Normalizer(), Resizer()]),
                                                event_k=parser.event_k)
            collater_fn = collater_raw

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, batch_size=parser.batch_size, num_workers=parser.num_worker, shuffle=True, collate_fn=collater_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=parser.num_worker, shuffle=True, collate_fn=collater_fn)

    # dataset_val1 = CSVDataset_event(train_file=f'{base_dir}/DSEC_detection_labels/events/labels_filtered_test.csv', class_list=parser.csv_classes,
    #                                 root_event_dir=parser.root_event,root_img_dir=parser.root_img, transform=transforms.Compose([Normalizer(), Resizer()]))
    # dataloader_val = DataLoader(dataset_val , batch_size=1, num_workers=1, shuffle=True, collate_fn=collater)

    # if dataset_val is not None:
    #     sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    #     dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    list_models = ['early_fusion','fpn_fusion', 'event', 'rgb', 'fpn_fusion_est']
    if parser.fusion in  list_models:
        if parser.depth == 50:
            retinanet = model.resnet50(num_classes=dataset_train.num_classes(), fusion_model=parser.fusion, pretrained=parser.pretrained)
        if parser.depth == 101:
            retinanet = model.resnet101(num_classes=dataset_train.num_classes(), fusion_model=parser.fusion, pretrained=parser.pretrained)
    else:
        raise ValueError('Unsupported model fusion')

    use_gpu = True
    if parser.continue_training:
        checkpoint = torch.load(parser.checkpoint)
        retinanet.load_state_dict(checkpoint['model_state_dict'])
        epoch_loss_all = checkpoint['loss']
        epoch_total = checkpoint['epoch']
        print('training sensor fusion model')
        retinanet.eval()
    else:
        epoch_total = 0
        epoch_loss_all =[]
    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=100)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    num_batches = 0
    start = time.time()
    # print('sensor fusion, impulse_noise images')
    # mAP = csv_eval.evaluate(dataset_val1, retinanet)
    print(time_since(start))
    epoch_loss = []


    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_total += 1

        # mAP = csv_eval.evaluate(dataset_test, retinanet, save_detection=False, save_folder=save_dir, save_path=save_dir)

        for iter_num, data in enumerate(tqdm(dataloader_train)):
            try:
                classification_loss, regression_loss = retinanet([data['img_rgb'],data['img'].cuda().float(),data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                num_batches += 1
                if num_batches == 8:  # optimize every 5 mini-batches
                    optimizer.step()
                    optimizer.zero_grad()
                    num_batches = 0


                loss_hist.append(float(loss))


                if iter_num % 500 ==0:
                    print(
                        '[sensor fusion homographic] [{}], Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            time_since(start), epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                    epoch_loss.append(np.mean(loss_hist))

                wandb_logger.wandb.log({'train/cls loss': classification_loss.detach()})
                wandb_logger.wandb.log({'train/reg loss': regression_loss.detach()})
                wandb_logger.wandb.log({'train/total loss': loss.detach()})

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        # if parser.dataset == 'coco':
        #
        #     print('Evaluating dataset')
        #
        #     coco_eval.evaluate_coco(dataset_val, retinanet)
        #
        # elif parser.dataset == 'csv' and parser.csv_val is not None:
        #
        #     print('Evaluating dataset')
        #
        #     mAP = csv_eval.evaluate(dataset_val, retinanet)

        ## evaluation ##
        # mAP = csv_eval.evaluate_coco_map(dataset_test, retinanet, save_detection=False, save_folder=save_dir,
        #                                  save_path=save_dir)

        if parser.event_type == 'raw':
            mAP = csv_eval.evaluate_coco_map(dataloader_test, retinanet, save_detection=False, save_folder=save_dir,
                                             save_path=save_dir, event_type=parser.event_type)
        else:
            mAP = csv_eval.evaluate_coco_map(dataset_test, retinanet, save_detection=False, save_folder=save_dir,
                                             save_path=save_dir, event_type=parser.event_type)

        wandb_logger.wandb.log({'test/person': np.mean(mAP[0])})
        wandb_logger.wandb.log({'test/large_vehicle': np.mean(mAP[1])})
        wandb_logger.wandb.log({'test/car': np.mean(mAP[2])})
        wandb_logger.wandb.log({'test/mAP': np.mean(np.array(mAP[0] + mAP[1] + mAP[2]) / 3)})

        scheduler.step(np.mean(epoch_loss))

        # torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))
        if epoch_num % 10 == 0:
            torch.save({'epoch': epoch_total, 'model_state_dict': retinanet.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': np.append(epoch_loss_all, epoch_loss)},
                         f'{save_dir}/{epoch_total}.pt') # f'{parser.dataset}_fpn_homographic_retinanet_retinanet101_{epoch_total}.pt'


    # retinanet.eval()

    torch.save({'epoch': epoch_total, 'model_state_dict': retinanet.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': np.append(epoch_loss_all,epoch_loss)}, f'{parser.dataset}_fpn_homographic_retinanet_retinanet101_{epoch_total}.pt')


if __name__ == '__main__':
    main()
