import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
from est.models import ESTNet
import random

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet_fpn_fusion_est(nn.Module):

    def __init__(self, num_classes, block, layers,
                 voxel_dimension, crop_dimension, mlp_layers, activation, value_layer, projection,
                 pretrained):
        self.inplanes = 64
        super(ResNet_fpn_fusion_est, self).__init__()
        # RGB branch
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #Event branch
        self.inplanes = 64
        self.conv1_event = nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1_event = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_event = nn.BatchNorm2d(64)
        self.relu_event = nn.ReLU(inplace=True)
        self.maxpool_event = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_event = self._make_layer(block, 64, layers[0])
        self.layer2_event = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_event = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_event = self._make_layer(block, 512, layers[3], stride=2)
        self.est_model = ESTNet(voxel_dimension, crop_dimension, num_classes, mlp_layers, activation, value_layer, projection, pretrained)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(2*fpn_sizes[0], 2*fpn_sizes[1], 2*fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def get_score(self, inputs, classification, regression, anchors):
        img_batch_rgb, img_batch_event = inputs

        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch_rgb)

        finalResult = [[], [], []]

        finalScores = torch.Tensor([])
        finalAnchorBoxesIndexes = torch.Tensor([]).long()
        finalAnchorBoxesCoordinates = torch.Tensor([])

        if torch.cuda.is_available():
            finalScores = finalScores.cuda()
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

        for i in range(classification.shape[2]):
            scores = torch.squeeze(classification[:, :, i])
            scores_over_thresh = (scores > 0.05)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

            finalResult[0].extend(scores[anchors_nms_idx])
            finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
            finalResult[2].extend(anchorBoxes[anchors_nms_idx])

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            if torch.cuda.is_available():
                finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


    def _forward_impl(self, inputs):

        # if self.training:
        #     img_batch_rgb, img_batch_event, annotations = inputs
        #     if random.uniform(0, 1) < 0.15:
        #         img_batch_rgb.zero_()
        # else:
        #     img_batch_rgb, img_batch_event = inputs
        img_batch_rgb, img_batch_event = inputs

        x_rgb = self.conv1(img_batch_rgb)
        x_rgb = self.bn1(x_rgb)
        x_rgb = self.relu(x_rgb)
        x_rgb = self.maxpool(x_rgb)

        x1_rgb = self.layer1(x_rgb)
        x2_rgb = self.layer2(x1_rgb)
        x3_rgb = self.layer3(x2_rgb)
        x4_rgb = self.layer4(x3_rgb)

        #Event stream
        vox_cropped = self.est_model._forward_impl(img_batch_event)
        x_event = self.conv1_event(vox_cropped)
        x_event = self.bn1_event(x_event)
        x_event = self.relu_event(x_event)
        x_event = self.maxpool_event(x_event)

        x1_event = self.layer1_event(x_event)
        x2_event = self.layer2_event(x1_event)
        x3_event = self.layer3_event(x2_event)
        x4_event = self.layer4_event(x3_event)

        x2 = torch.cat((x2_event,x2_rgb),1)
        x3 = torch.cat((x3_event, x3_rgb), 1)
        x4 = torch.cat((x4_event, x4_rgb), 1)
        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch_rgb)

        return classification, regression, anchors

        # if self.training:
        #     return self.focalLoss(classification, regression, anchors, annotations)
        # else:
        #     transformed_anchors = self.regressBoxes(anchors, regression)
        #     transformed_anchors = self.clipBoxes(transformed_anchors, img_batch_rgb)
        #
        #     finalResult = [[], [], []]
        #
        #     finalScores = torch.Tensor([])
        #     finalAnchorBoxesIndexes = torch.Tensor([]).long()
        #     finalAnchorBoxesCoordinates = torch.Tensor([])
        #
        #     if torch.cuda.is_available():
        #         finalScores = finalScores.cuda()
        #         finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
        #         finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()
        #
        #     for i in range(classification.shape[2]):
        #         scores = torch.squeeze(classification[:, :, i])
        #         scores_over_thresh = (scores > 0.05)
        #         if scores_over_thresh.sum() == 0:
        #             # no boxes to NMS, just continue
        #             continue
        #
        #         scores = scores[scores_over_thresh]
        #         anchorBoxes = torch.squeeze(transformed_anchors)
        #         anchorBoxes = anchorBoxes[scores_over_thresh]
        #         anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
        #
        #         finalResult[0].extend(scores[anchors_nms_idx])
        #         finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
        #         finalResult[2].extend(anchorBoxes[anchors_nms_idx])
        #
        #         finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
        #         finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
        #         if torch.cuda.is_available():
        #             finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()
        #
        #         finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
        #         finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
        #
        #     return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]



class AdvResNet_fpn_fusion_est(ResNet_fpn_fusion_est):
    def __init__(self, num_classes, block, layers,
                 voxel_dimension=(9, 480, 640),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(480, 640),  # dimension of crop before it goes into classifier
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 value_layer="ValueLayer",
                 projection=None,
                 pretrained=True,
                 adv=False,
                 adv_test=False,
                 attack_mode='shifting_event'):

        self.inplanes = 64
        super(AdvResNet_fpn_fusion_est, self).__init__(num_classes, block, layers,
                                                       voxel_dimension, crop_dimension, mlp_layers, activation, value_layer, projection, pretrained)
        self.adv = adv
        self.adv_test = adv_test
        self.attack_mode = attack_mode

    def set_attacker(self, attacker):
        self.attacker = attacker

    def forward(self, inputs):
        training = self.training

        img_batch_rgb, img_batch_event, labels = inputs

        if training:
            if random.uniform(0, 1) < 0.15:
                img_batch_rgb.zero_()

            if self.adv == True:
                self.eval()
                adv_event, _ = self.attacker.attack(img_batch_rgb, img_batch_event, labels, self, mode=self.attack_mode)
                with torch.no_grad():
                    adv_event[:, 4] += (img_batch_event[:, -1].max() + 1)  # adv batch_size
                    img_batch_event = torch.cat([img_batch_event, adv_event], dim=0)
                    labels = torch.cat([labels, labels], dim=0)
                    img_batch_rgb = torch.cat([img_batch_rgb, img_batch_rgb], dim=0)
                    x = (img_batch_rgb, img_batch_event)
                self.train()
                classification, regression, anchors = self._forward_impl(x)
                return classification, regression, anchors, labels
            else:
                self.train()
                classification, regression, anchors = self._forward_impl((img_batch_rgb, img_batch_event))
                return classification, regression, anchors, labels
        else:
            # adv_test
            if self.adv_test == True:
                adv_event, _ = self.attacker.attack(img_batch_rgb, img_batch_event, labels, self, mode=self.attack_mode)
                with torch.no_grad():
                    classification, regression, anchors = self._forward_impl((img_batch_rgb, img_batch_event))
                    adv_classification, adv_regression, adv_anchor = self._forward_impl((img_batch_rgb, adv_event))
                    pred = (classification, regression, anchors)
                    adv_pred = (adv_classification, adv_regression, adv_anchor) # TODO: check if this is correct

                return pred, adv_pred
            else:
                classification, regression, anchors = self._forward_impl((img_batch_rgb, img_batch_event))
                scores, labels, boxes = self.get_score((img_batch_rgb, img_batch_event), classification, regression, anchors)

                return scores, labels, boxes




class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes) # [B, width * height * num_anchors, num_classes)


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
