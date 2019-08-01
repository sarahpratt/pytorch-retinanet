from collections import defaultdict
import pdb
from PIL import Image
import numpy as np
import cv2

class BboxEval:
    def __init__(self):
        self.per_verb_occ_bboxes = defaultdict(float)
        self.per_verb_all_correct_bboxes = defaultdict(float)
        self.per_verb_roles_bboxes = defaultdict(float)
        self.per_verb_roles_correct_bboxes = defaultdict(float)

        self.per_verb_occ = defaultdict(float)
        self.per_verb_all_correct = defaultdict(float)
        self.per_verb_roles = defaultdict(float)
        self.per_verb_roles_correct = defaultdict(float)

        self.all_verbs = 0.0
        self.correct_verbs = 0.0


    def verb(self):
        return self.correct_verbs/self.all_verbs


    def value_all(self):
        sum_value_all = 0.0
        total_value_all = 0.0
        for verb in self.per_verb_occ:
            sum_value_all += float(self.per_verb_all_correct[verb])/float(self.per_verb_occ[verb])
            total_value_all += 1.0
        return sum_value_all/total_value_all


    def value(self):
        sum_value = 0.0
        total_value = 0.0
        for verb in self.per_verb_roles:
            sum_value += float(self.per_verb_roles_correct[verb]) / float(self.per_verb_roles[verb])
            total_value += 1.0
        return sum_value / total_value


    def value_all_bbox(self):
        sum_value_all = 0.0
        total_value_all = 0.0
        for verb in self.per_verb_occ_bboxes:
            sum_value_all += float(self.per_verb_all_correct_bboxes[verb])/float(self.per_verb_occ_bboxes[verb])
            total_value_all += 1.0
        return sum_value_all/total_value_all


    def value_bbox(self):
        sum_value = 0.0
        total_value = 0.0
        for verb in self.per_verb_roles_bboxes:
            sum_value += float(self.per_verb_roles_correct_bboxes[verb]) / float(self.per_verb_roles_bboxes[verb])
            total_value += 1.0
        return sum_value / total_value


    def update(self, pred_verb, pred_nouns, pred_bboxes, gt_verb, gt_nouns, gt_bboxes, verb_order):
        order = verb_order[gt_verb]["order"]
        #order = ["agent", "tool"]

        self.all_verbs += 1.0
        self.per_verb_occ[gt_verb] += 1.0
        self.per_verb_occ_bboxes[gt_verb] += 1.0

        self.per_verb_roles[gt_verb] += len(order)
        self.per_verb_roles_bboxes[gt_verb] += len(order)

        if len(pred_nouns) == 0:
            pdb.set_trace()

        if pred_verb == gt_verb:
            self.correct_verbs += 1.0
            value_all_bbox = 1.0
            value_all = 1.0
            for i in range(len(order)):
                if pred_nouns[i] in gt_nouns[i]:
                    self.per_verb_roles_correct[gt_verb] += 1.0
                else:
                    value_all = 0.0
                if pred_nouns[i] in gt_nouns[i] and (self.bb_intersection_over_union(pred_bboxes[i], gt_bboxes[i])):
                    self.per_verb_roles_correct_bboxes[gt_verb] += 1.0
                else:
                    value_all_bbox = 0.0
            self.per_verb_all_correct_bboxes[gt_verb] += value_all_bbox
            self.per_verb_all_correct[gt_verb] += value_all


    def visualize(self, pred_verb, pred_nouns, pred_bboxes, gt_verb, gt_nouns, gt_bboxes, verb_order, image, word_dict):
        verb_order = verb_order[gt_verb]["order"]

        img = Image.open('./images_512/' + image)
        img = np.float32(img)

        width = img.shape[1]

        for b in pred_bboxes:
            #pdb.set_trace()
            if b is not None and b[0] != -1:
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 3)

        img_gt = Image.open('./images_512/' + image)
        img_gt = np.float32(img_gt)
        for b in gt_bboxes:
            if b is not None and b[0] != -1:
                cv2.rectangle(img_gt, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 3)

        new_im = Image.new('RGB', (img.shape[1] * 2 + 200 * 2, img.shape[0]))
        new_im = np.float32(new_im)

        if len(gt_nouns) == 0:
            pdb.set_trace()

        cv2.putText(new_im, gt_verb, (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        for i in range(len(verb_order)):
            cv2.putText(new_im, verb_order[i], (0, 60*(i+1)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
            j = 0
            for word in gt_nouns[i]:
                if word in word_dict:
                    w = word_dict[word]['gloss'][0]
                else:
                    w = word
                cv2.putText(new_im, w, (0, 80*(i+1) + j*20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)
                j += 1

        cv2.putText(new_im, pred_verb, (img.shape[1] + 200, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        for i in range(len(verb_order)):
            cv2.putText(new_im, verb_order[i], (img.shape[1] + 200, 60*(i+1)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
            j = 0
            #pdb.set_trace()
            #for word in pred_nouns[i]:
            if pred_nouns[i] in word_dict:
                w = word_dict[pred_nouns[i]]['gloss'][0]
            else:
                w = pred_nouns[i]
            cv2.putText(new_im, w, (img.shape[1] + 200, 80*(i+1) + j * 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)
            j += 1

        new_im = Image.fromarray(np.uint8(new_im))
        img = Image.fromarray(np.uint8(img))
        img_gt = Image.fromarray(np.uint8(img_gt))
        new_im.paste(img_gt, box=(200, 0))
        new_im.paste(img, box=(400 + width, 0))
        return new_im

        #new_im.save('./predictions/' + image)


    def visualize_for_demo(self, b, image, color):

        img = Image.open('./images_512/' + image)
        img = np.float32(img)

        if b is not None and b[0] != -1:
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 6)

        img = Image.fromarray(np.uint8(img))
        return img



    def bb_intersection_over_union(self, boxA, boxB):
        if boxA is None and boxB is None:
            return True
        if boxA is None or boxB is None:
            return False
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        if iou > 0.5:
            return True
        else:
            return False