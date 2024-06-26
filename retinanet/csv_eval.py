from __future__ import print_function

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
import pickle
from tqdm import tqdm


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            try:
                scale = data['scales']
            except:
                scale = 1

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = retinanet([data['img_rgb'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0), torch.tensor(data['img']).permute(2, 0, 1).cuda().float().unsqueeze(dim=0)])
                # scores, labels, boxes = retinanet(torch.tensor(data['img']).permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()

            # correct boxes for image scale
            # print(f'boxes:{boxes}')
            # print(f'labels:{scale}')
            boxes /= scale
            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_detections2(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None, event_type='voxel'):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()

    with torch.no_grad():

        for index, data in enumerate(dataset):
            try:
                scale = data['scales']
            except:
                scale = 1

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = retinanet([data['img_rgb'].cuda().float(), data['img'].cuda().float()])
                # scores, labels, boxes = retinanet(torch.tensor(data['img']).permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # correct boxes for image scale
            # print(f'boxes:{boxes}')
            # print(f'labels:{scale}')
            boxes /= scale
            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate(
                    [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_detections3(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()

    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]

            # event 1 batched processing (changed)
            event = data['img']
            event = np.concatenate([event, 0 * np.ones((len(event), 1), dtype=np.float32)], 1)

            try:
                scale = data['scales']
            except:
                scale = 1

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = retinanet([data['img_rgb'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0),
                                                   torch.tensor(event).cuda().float(),
                                                   data['annot']])
                # scores, labels, boxes = retinanet(torch.tensor(data['img']).permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # correct boxes for image scale
            # print(f'boxes:{boxes}')
            # print(f'labels:{scale}')
            boxes /= scale
            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate(
                    [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def _get_annotations2(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.dataset.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.dataset.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.dataset.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations



def evaluate_coco_map(
    generator,
    retinanet,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_detection = True,
    save_folder = './',
    load_detection = False,
    save_path=None,
    event_type='voxel',
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save precision recall curve of each label.
    # Returns
        A dict mapping class names to mAP scores.
    """



    # gather all detections and annotations
    detections_file = os.path.join(save_folder,'detections.txt')
    annotations_file = os.path.join(save_folder,'annotations.txt')

    if load_detection == True:
        with open(detections_file, "rb") as fp:  
            all_detections = pickle.load(fp)

        with open(annotations_file, "rb") as fp: 
            all_annotations = pickle.load(fp)

    else:
        if event_type == 'raw':
            # all_detections = _get_detections2(generator, retinanet,
            #                                  score_threshold=score_threshold, max_detections=max_detections,
            #                                  save_path=save_path, event_type=event_type)
            # all_annotations = _get_annotations2(generator)
            #
            # generator = generator.dataset

            all_detections = _get_detections3(generator, retinanet,
                                             score_threshold=score_threshold, max_detections=max_detections,
                                             save_path=save_path)
            all_annotations = _get_annotations(generator)

        else:
            all_detections = _get_detections(generator, retinanet,
                                                 score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
            all_annotations = _get_annotations(generator)
        if save_detection == True:
            with open(detections_file, "wb") as fp:
                pickle.dump(all_detections, fp)

            with open(annotations_file, "wb") as fp:
                pickle.dump(all_annotations , fp)


    average_precisions = {}
    average_precisions_coco = {}
    for label in range(generator.num_classes()):
        average_precisions_coco[label] = []

    iou_values = np.arange(0.5, 1.00, 0.05).tolist()

    for label in range(generator.num_classes()):
        false_positives = []
        true_positives  = []
        for idx,iou_threshold1 in enumerate(iou_values):
            false_positives.append(np.zeros((0,)))
            true_positives.append( np.zeros((0,)))

        scores= np.zeros((0,))
        num_annotations = 0.0

        
        for i in range(len(generator)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []
            for idx, iou_threshold1 in enumerate(iou_values):
                detected_annotations.append([])

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    for idx,iou_threshold1 in enumerate(iou_values):
                        false_positives[idx] = np.append(false_positives[idx], 1)
                        true_positives[idx]  = np.append(true_positives[idx], 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                for idx,iou_threshold1 in enumerate(iou_values):
                    if max_overlap >= iou_threshold1 and assigned_annotation not in detected_annotations[idx]:
                        false_positives[idx] = np.append(false_positives[idx], 0)
                        true_positives[idx]  = np.append(true_positives[idx], 1)
                        detected_annotations[idx].append(assigned_annotation)
                    else:
                        false_positives[idx] = np.append(false_positives[idx], 1)
                        true_positives[idx]  = np.append(true_positives[idx], 0)


        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            average_precisions_coco[label].append(average_precision)
            continue

        # sort by score
        indices         = np.argsort(-scores)
        for idx,iou_threshold1 in enumerate(iou_values):
            false_positives[idx] = false_positives[idx][indices]
            true_positives[idx]  = true_positives[idx][indices]

            # compute false positives and true positives
            false_positives[idx] = np.cumsum(false_positives[idx])
            true_positives[idx]  = np.cumsum(true_positives[idx])

            # compute recall and precision
            recall    = true_positives[idx] / num_annotations
            precision = true_positives[idx] / np.maximum(true_positives[idx] + false_positives[idx], np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = _compute_ap(recall, precision)
            average_precisions[label] = average_precision, num_annotations
            average_precisions_coco[label].append(average_precision)


        # print('\nmAP:')
        label_name = generator.label_to_name(label)
        # print(len(average_precisions_coco[label]))
        # print('mAP {}: {}'.format(label_name, mean(average_precisions_coco[label]) ))
        # print('mAP50 {}: {}'.format(label_name, average_precisions_coco[label][0]))
        # print('mAP75 {}: {}'.format(label_name, average_precisions_coco[label][5]))
        # print("Precision: ",precision[-1])
        # print("Recall: ",recall[-1])
        
        if save_path!=None:
            plt.plot(recall,precision)
            # naming the x axis 
            plt.xlabel('Recall') 
            # naming the y axis 
            plt.ylabel('Precision') 

            # giving a title to my graph 
            plt.title('Precision Recall curve') 

            # function to show the plot
            plt.savefig(save_path+'/'+label_name+'_precision_recall.jpg')



    return average_precisions_coco



def evaluate(
    generator,
    retinanet,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_detection = True,
    save_folder = './',
    load_detection = False,
    save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save precision recall curve of each label.
    # Returns
        A dict mapping class names to mAP scores.
    """



    # gather all detections and annotations
    detections_file = os.path.join(save_folder,'detections.txt')
    annotations_file = os.path.join(save_folder,'annotations.txt')

    if load_detection == True:
        with open(detections_file, "rb") as fp:  
            all_detections = pickle.load(fp)

        with open(annotations_file, "rb") as fp: 
            all_annotations = pickle.load(fp)

    else:
        all_detections     = _get_detections(generator, retinanet, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
        all_annotations    = _get_annotations(generator)
        if save_detection == True:
            with open(detections_file, "wb") as fp:
                pickle.dump(all_detections, fp)

            with open(annotations_file, "wb") as fp:
                pickle.dump(all_annotations , fp)


    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations


        # print('\nmAP:')
        label_name = generator.label_to_name(label)
        # print('{}: {}'.format(label_name, average_precisions[label][0]))
        # print("Precision: ",precision[-1])
        # print("Recall: ",recall[-1])
        
        if save_path!=None:
            plt.plot(recall,precision)
            # naming the x axis 
            plt.xlabel('Recall') 
            # naming the y axis 
            plt.ylabel('Precision') 

            # giving a title to my graph 
            plt.title('Precision Recall curve') 

            # function to show the plot
            plt.savefig(save_path+'/'+label_name+'_precision_recall.jpg')



    return average_precisions

