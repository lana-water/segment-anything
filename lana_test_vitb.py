from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import build_sam, SamPredictor
import cv2, numpy as np
import numpy as np
import torch
import matplotlib.pyplot as plt

predictor = SamPredictor(build_sam(checkpoint="/Users/mac/Documents/GitHub/segment-anything/sam_vit_b_01ec64.pth"))

img = cv2.imread("/Users/mac/Documents/GitHub/segment-anything/composite_img.jpg")
# print(img, "end")

# Resize the image to a smaller resolution
# scale_factor = 0.05
# new_width = int(img.shape[1] * scale_factor)
# new_height = int(img.shape[0] * scale_factor)
# image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
image = cv2.resize(img, None, fx = 0.2, fy = 0.2)
# print(image.shape)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# A = np.squeeze(np.asarray(img))
predictor.set_image(image)
# print(predictor)

sam = sam_model_registry["vit_b"](checkpoint="/Users/mac/Documents/GitHub/segment-anything/sam_vit_b_01ec64.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 