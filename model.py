import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors
import losses
# from lib.nms.pth_nms import pth_nms
import torch.nn.functional as F
import pdb


# def nms(dets, thresh):
#     "Dispatch to either CPU or GPU NMS implementations.\
#     Accept dets as tensor"""
#
#     return pth_nms(dets, thresh)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


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
        # return [P4_x, P5_x, P6_x, P7_x]


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
        out1 = self.act4(out)
        out = self.output(out1)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.spatial_conv = nn.Conv2d(2, 64, kernel_size=1)
        self.bbox_conv = nn.Conv2d(4, 64, kernel_size=1)
        self.mask_conv = nn.Conv2d(1, 64, kernel_size=1)

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        #self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.features_linear = nn.Linear(feature_size, 1)
        # self.act_binary = nn.Sigmoid()
        #self.output_retina = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_retina = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=1)

        self.output_act_retina = nn.Sigmoid()

    def forward(self, x):

        x_directions = (torch.arange(x.shape[2]).float() / x.shape[2]).cuda()
        x_directions = x_directions.view(1, 1, x_directions.shape[0], 1).expand([x.shape[0], 1, x.shape[2], x.shape[3]])

        y_directions = (torch.arange(x.shape[3]).float() / x.shape[3]).cuda()
        y_directions = y_directions.view(1, 1, 1, y_directions.shape[0]).expand([x.shape[0], 1, x.shape[2], x.shape[3]])
        spatial = torch.cat((x_directions, y_directions), dim=1)
        spatial = self.spatial_conv(spatial)

        #bbox = bbox.view(x.shape[0], 4, 1, 1).expand([x.shape[0], 4, x.shape[2], x.shape[3]])

        #bbox = self.bbox_conv(bbox)
        #mask = self.mask_conv(w)

        #new_x = torch.cat((x, spatial, bbox, mask), dim=1)
        new_x = torch.cat((x, spatial), dim=1)


        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        #out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        #out = self.bn4(out)
        # BBox Binary Logit
        bbox_exists = self.pool(out).squeeze()
        bbox_exists = self.features_linear(bbox_exists)

        # Classification Branch
        out = self.act4(out)
        out1 = self.output_retina(out)
        out1 = self.output_act_retina(out1)
        out1 = out1.permute(0, 2, 3, 1)  # out is B x C x W x H, with C = n_classes + n_anchors
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        #nouns = nouns.view(x.shape[0], 1, self.num_classes)

        #nouns = F.softmax(nouns.view(x.shape[0], 1, self.num_classes), dim=2)
        #return F.softmax(out2.contiguous().view(x.shape[0], -1, self.num_classes), dim=2)*nouns, bbox_exists
        #return out2.contiguous().view(x.shape[0], -1, self.num_classes), bbox_exists
        return out2.contiguous().view(new_x.shape[0], -1, self.num_classes), bbox_exists

class LearnableVector(nn.Module):

    def __init__(self, n):
        super(LearnableVector, self).__init__()
        self.weight = nn.Parameter(torch.randn(1,n))

    def forward(self, x):
        return self.weight * x


class ClassifyLocalFeatures(nn.Module):
    def __init__(self, vocab_size):
        super(ClassifyLocalFeatures, self).__init__()
        self.local_linear = nn.Linear(2048, 256)
        self.features_linear = nn.Linear(512, 256)
        self.act = nn.ReLU()
        self.vocab_linear = nn.Linear(256, vocab_size)

    def forward(self, local_features, rnn_features):
        local = self.local_linear(local_features)
        out = torch.cat((local, rnn_features), dim=1)
        out = self.features_linear(out)
        out = self.act(out)
        return self.vocab_linear(out)


class ResNet_RetinaNet_RNN(nn.Module):

    def __init__(self, num_classes, num_nouns, block, layers, cat_features=False):
        self.inplanes = 64
        super(ResNet_RetinaNet_RNN, self).__init__()

        self.num_classes = num_classes
        self.num_nouns = num_nouns

        self._init_resnet(block, layers)
        self.fpn = PyramidFeatures(self.fpn_sizes[0], self.fpn_sizes[1], self.fpn_sizes[2])

        self.regressionModel = RegressionModel(768)
        self.classificationModel = ClassificationModel(768, num_classes=num_classes)
            # self.regressionModel = RegressionModel(256)
            # self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        #self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()
        self.cat_features = cat_features

        self.use_expert = False

        self.noun_classifier = ClassifyLocalFeatures(num_nouns)
        #pdb.set_trace()

        #self.noun_attn = LearnableVector(num_classes)

        self._convs_and_bn_weights()

        # verb predictor
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 504)

        # init embeddings
        self.verb_embeding = nn.Embedding(504, 512)
        self.noun_embedding = nn.Embedding(num_nouns, 512)
        self.bbox_width_embed = nn.Embedding(11, 16)
        self.bbox_height_embed = nn.Embedding(11, 16)
        self.bbox_x_embed = nn.Embedding(11, 16)
        self.bbox_y_embed = nn.Embedding(11, 16)

        self.rnn = nn.LSTMCell(2048 + 512 + 64, 1024*2)

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        self.rnn_linear = nn.Linear(1024*2, 256)
        self.noun_fc = nn.Linear(1024*2, num_classes)

        # fill class/reg branches with weights
        prior = 0.01

        self.classificationModel.output_retina.weight.data.fill_(0)
        self.classificationModel.output_retina.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()
        self.loss_function = nn.CrossEntropyLoss()
        self.all_box_regression = False


    def _init_resnet(self, block, layers):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            self.fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels,
                              self.layer3[layers[2] - 1].conv2.out_channels,
                              self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            self.fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels,
                              self.layer3[layers[2] - 1].conv3.out_channels,
                              self.layer4[layers[3] - 1].conv3.out_channels]


    def _convs_and_bn_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def get_local_visual_features(self, featues, bbox, no_grounding, batch_size):
        x_start_bin = self.get_bin(featues.shape[2], bbox[:, 0])
        end_x_bin = self.get_bin(featues.shape[2], bbox[:, 1])
        start_y_bin = self.get_bin(featues.shape[3], bbox[:, 2])
        end_y_bin = self.get_bin(featues.shape[3], bbox[:, 3])
        local_features = torch.zeros(batch_size, 2048).cuda()
        for i in range(batch_size):
            if not no_grounding[i]:
                #feature_slice = featues[i, :, x_start_bin[i]:end_x_bin[i]+1, start_y_bin[i]:end_y_bin[i]+1]
                i_bbox = [bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]]
                weights = self.get_weights(featues.shape[2], featues.shape[3], x_start_bin[i], end_x_bin[i], start_y_bin[i], end_y_bin[i], i_bbox, batch_size)
                local_features[i] = self.avgpool(featues[i, :]*weights).squeeze()
        return local_features


    def get_all_weights(self, featues, bbox, no_grounding, batch_size):
        x_start_bin = self.get_bin(featues.shape[2], bbox[:, 0])
        end_x_bin = self.get_bin(featues.shape[2], bbox[:, 1])
        start_y_bin = self.get_bin(featues.shape[3], bbox[:, 2])
        end_y_bin = self.get_bin(featues.shape[3], bbox[:, 3])
        all_weights = torch.zeros(batch_size, 1, featues.shape[2], featues.shape[3]).cuda()
        for i in range(batch_size):
            if not no_grounding[i]:
                i_bbox = [bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]]
                weights = self.get_weights(featues.shape[2], featues.shape[3], x_start_bin[i], end_x_bin[i], start_y_bin[i], end_y_bin[i], i_bbox, batch_size)
                all_weights[i] = weights
        return all_weights


    def get_bin(self, image_size, location_percentage):
        location_percentage[location_percentage >= 1.0] = .99
        bin = torch.floor(image_size * location_percentage).long()
        bin[bin < 0] = 0
        return bin.long()


    def get_weights(self, feature_length, feature_height, x_start_bin, end_x_bin, start_y_bin, end_y_bin, bbox, batch_size):
        x_weights = torch.zeros(1, feature_length, feature_height).cuda()
        x_weights[:, x_start_bin+1:end_x_bin, :] = 1.0
        x_weights[:, x_start_bin, :] = (x_start_bin+1) - feature_length*bbox[0]
        x_weights[:, end_x_bin, :] = feature_length * bbox[1] - end_x_bin

        y_weights = torch.zeros(1, feature_length, feature_height).cuda()
        y_weights[:, :, start_y_bin + 1:end_y_bin] = 1.0
        y_weights[:, :, start_y_bin] = (start_y_bin + 1) - feature_height * bbox[2]
        y_weights[:, :, end_y_bin] = feature_height * bbox[3] - end_y_bin

        return x_weights*y_weights


    def forward(self, inputs, roles, detach_resnet=False, use_gt_nouns=False, use_gt_verb=False):

        if self.training:
            img_batch, annotations, verb, widths, heights = inputs
        else:
            img_batch, verb, widths, heights = inputs

        #pdb.set_trace()

        batch_size = img_batch.shape[0]

        # Extract Visual Features
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)

        if detach_resnet:
            x2 = self.layer2(x1).detach()
            x3 = self.layer3(x2).detach()
            x4 = self.layer4(x3).detach()
        else:
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)

        image_predict = self.avgpool(x4)
        image_predict = image_predict.squeeze()
        if len(image_predict.shape) == 1:
            image_predict = image_predict.unsqueeze(0)

        verb_predict = self.fc(image_predict)
        verb_guess = torch.argmax(verb_predict, dim=1)
        verb_loss = self.loss_function(verb_predict, verb)
        if self.training or use_gt_verb:
            previous_word = self.verb_embeding(verb.long())
        else:
            previous_word = self.verb_embeding(verb_guess)

        # Get feature pyramid
        features = self.fpn([x2, x3, x4])
        anchors = self.anchors(img_batch)
        features.pop(0)  # SARAH - remove feature batch

        # init LSTM inputs
        hx, cx = torch.zeros(batch_size, 1024*2).cuda(x.device), torch.zeros(batch_size, 1024*2).cuda(x.device)
        previous_box_embed = torch.zeros(batch_size, 64).cuda(x.device)
        bbox = torch.zeros(batch_size, 4).cuda(x.device)


        # init losses
        all_class_loss = 0
        all_bbox_loss = 0
        all_reg_loss = 0

        if self.training:
            class_list = []
            reg_list = []
            bbox_pred_list = []
        else:
            noun_predicts = []
            bbox_predicts = []
            bbox_exist_list = []

        noun_loss = 0.0

        previous_location_features = torch.zeros(batch_size, 2048).cuda(x.device)

        for i in range(6):
            rnn_input = torch.cat((image_predict, previous_word, previous_box_embed), dim=1)
            hx, cx = self.rnn(rnn_input, (hx, cx))
            rnn_output = self.rnn_linear(hx)

            just_rnn = [rnn_output.view(batch_size, 256, 1, 1).expand(feature.shape) for feature in features]
            rnn_feature_mult = [rnn_output.view(batch_size, 256, 1, 1).expand(feature.shape) * feature for feature in features]
            rnn_feature_shapes = [torch.cat([just_rnn[ii], rnn_feature_mult[ii], features[ii]], dim=1) for ii in range(len(features))]

            regression = torch.cat([self.regressionModel(rnn_and_features) for rnn_and_features in rnn_feature_shapes], dim=1)
            classifications = []
            bbox_exist = []

            for ii in range(len(rnn_feature_shapes)):
                classication = self.classificationModel(rnn_feature_shapes[ii])
                bbox_exist.append(classication[1])
                classifications.append(classication[0])

            if len(bbox_exist[0].shape) == 1:
                bbox_exist = [c.unsqueeze(0) for c in bbox_exist]

            bbox_exist = torch.cat([c for c in bbox_exist], dim=1)
            bbox_exist = torch.max(bbox_exist, dim=1)[0]

            # get max from K x A x W x H to get max classificiation and bbox
            classification = torch.cat([c for c in classifications], dim=1)
            best_per_box = torch.max(classification, dim=2)[0]
            best_bbox = torch.argmax(best_per_box, dim=1)
            class_boxes = classification[torch.arange(batch_size), best_bbox, :]
            classification_guess = torch.argmax(class_boxes, dim=1)


            if self.training:
                previous_boxes = annotations[:, i, :4]
            else:
                transformed_anchors = self.regressBoxes(anchors, regression)
                transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
                previous_boxes = transformed_anchors[torch.arange(batch_size), best_bbox, :]

            prev_heights = (previous_boxes[:, 2] - previous_boxes[:, 0]) / img_batch.shape[3]
            prev_widths = (previous_boxes[:, 3] - previous_boxes[:, 1]) / img_batch.shape[2]
            prev_ctr_y = previous_boxes[:, 0] / img_batch.shape[3] + 0.5 * prev_heights
            prev_ctr_x = previous_boxes[:, 1] / img_batch.shape[2] + 0.5 * prev_widths

            perc_start_x = previous_boxes[:, 1] / img_batch.shape[2]
            perc_end_x = previous_boxes[:, 3] / img_batch.shape[2]
            perc_start_y = previous_boxes[:, 0] / img_batch.shape[3]
            perc_end_y = previous_boxes[:, 2] / img_batch.shape[3]

            bbox = torch.stack([perc_start_x, perc_end_x, perc_start_y, perc_end_y], dim=1)

            if self.training:
                bbox[previous_boxes[:, 0] == -1] = -1.0
            else:
                bbox[bbox_exist < 0.5] = -1.0

            if self.training:
                previous_location_features = self.get_local_visual_features(x4, bbox, previous_boxes[:, 0] == -1,
                                                                            batch_size)
            else:
                previous_location_features = self.get_local_visual_features(x4, bbox, bbox_exist < 0.5, batch_size)

            noun_distribution = self.noun_classifier(previous_location_features, rnn_output)

            if self.training:
                gt = torch.zeros(batch_size, self.num_nouns).cuda(x.device)
                gt[torch.arange(batch_size), annotations[:, i, -1].long()] = 1
                gt[torch.arange(batch_size), annotations[:, i, -2].long()] = 1
                gt[torch.arange(batch_size), annotations[:, i, -3].long()] = 1
                noun_loss += F.binary_cross_entropy_with_logits(noun_distribution, gt.float())

            noun_guess = torch.argmax(noun_distribution, dim=1)

            if self.training and use_gt_nouns:
                ground_truth_1 = self.noun_embedding(annotations[:, i, -1].long())
                ground_truth_2 = self.noun_embedding(annotations[:, i, -2].long())
                ground_truth_3 = self.noun_embedding(annotations[:, i, -3].long())
                previous_word = torch.stack([ground_truth_1, ground_truth_2, ground_truth_3]).mean(dim=0)
            else:
                previous_word = self.noun_embedding(noun_guess)

            prev_widths = torch.ceil(prev_widths * 10).long()
            prev_heights = torch.ceil(prev_heights * 10).long()
            prev_ctr_x = torch.ceil(prev_ctr_x * 10).long()
            prev_ctr_y = torch.ceil(prev_ctr_y * 10).long()

            prev_widths = torch.clamp(prev_widths, 0, 10)
            prev_heights = torch.clamp(prev_heights, 0, 10)
            prev_ctr_x = torch.clamp(prev_ctr_x, 0, 10)
            prev_ctr_y = torch.clamp(prev_ctr_y, 0, 10)

            if not self.training:
                bbox_exist = torch.sigmoid(bbox_exist)
                prev_widths[bbox_exist < 0.5] = 0
                prev_heights[bbox_exist < 0.5] = 0
                prev_ctr_x[bbox_exist < 0.5] = 0
                prev_ctr_y[bbox_exist < 0.5] = 0

            previous_box_embed = torch.cat(
                [self.bbox_width_embed(prev_widths), self.bbox_height_embed(prev_heights),
                 self.bbox_x_embed(prev_ctr_x), self.bbox_y_embed(prev_ctr_y)], dim=1)

            if self.training:
                class_list.append(classification)
                reg_list.append(regression)
                bbox_pred_list.append(bbox_exist)
            else:
                bbox_predicts.append(previous_boxes)
                noun_predicts.append(noun_guess)
                bbox_exist_list.append(bbox_exist)

        if self.training:
            anns = annotations[:, :, :].unsqueeze(1)

            classification_all = torch.cat([c.unsqueeze(1) for c in class_list], dim=1)
            regression_all = torch.cat([c.unsqueeze(1) for c in reg_list], dim=1)
            bbox_exist_all = torch.cat([c.unsqueeze(1) for c in bbox_pred_list], dim=1)


            class_loss, reg_loss, bbox_loss = self.focalLoss(classification_all, regression_all, anchors, bbox_exist_all, anns.squeeze())
            all_class_loss += class_loss
            all_reg_loss += reg_loss
            all_bbox_loss += bbox_loss

            return all_class_loss, all_reg_loss, verb_loss, all_bbox_loss, noun_loss
        else:
            if use_gt_verb:
                return verb, noun_predicts, bbox_predicts, bbox_exist_list
            return verb_guess, noun_predicts, bbox_predicts, bbox_exist_list


def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_RetinaNet_RNN(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_RetinaNet_RNN(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, num_nouns, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #model = ResNet_RetinaNet_RNN(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs, cat_features=True)
    model = ResNet_RetinaNet_RNN(num_classes, num_nouns, Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'], model_dir='.')
        print("state dict")
        x = nn.Linear(2048, 504)
        state_dict['fc.weight'] = x.weight
        state_dict['fc.bias'] = x.bias
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_RetinaNet_RNN(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_RetinaNet_RNN(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model