import numpy as np
import json
import pdb

def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_

def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


with open('../ImSitu/train.json') as f:
    all = json.load(f)

scaled_boxes = []
for image in all:
    bboxes = all[image]['bb']
    for b in bboxes:
        box = bboxes[b]
        if box[0] != -1:
            width = float(box[2] - box[0])
            height = float(box[3] - box[1])
            if width != 0 and height != 0:
                height = height/width
                width = 1
                scaled_boxes.append([width,height])


clusters = kmeans(np.array(scaled_boxes), 2)
print(clusters)
print()