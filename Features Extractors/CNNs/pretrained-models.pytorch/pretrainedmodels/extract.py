import sys
import os
import numpy as np
import natsort
import torch
import pretrainedmodels
import pretrainedmodels.utils as utils

# python extract.py path_dataset output.npy

def extract_features(path_img):
    # LOAD IMAGE
    # transformations depending on the model
    # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
    tf_img = utils.TransformImage(model)
    load_img = utils.LoadImage()
    input_img = load_img(path_img)
    input_tensor = tf_img(input_img) # 3x400x225 -> 3x299x299 size may differ
    input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
    input = torch.autograd.Variable(input_tensor,
                                    requires_grad=False)
    # EXTRACT
    features = model(input) # obtain features
    features = features.data.cpu().numpy().tolist()[0]
    return features

# LOAD MODEL
model_name = sys.argv[2]
model = pretrainedmodels.__dict__[model_name](num_classes=1000,
                                              pretrained='imagenet')
model.eval()
# select output layer (obtain the layer before the classification)
model.last_linear = pretrainedmodels.utils.Identity()
torch.set_num_threads(4)

features = []
imgs_path = sys.argv[1]
images = natsort.natsorted(os.listdir(imgs_path))
print(model_name)
f = open("list.txt", "w+")
for i, img in enumerate(images):
    print(img, file=f)
    img = os.path.join(imgs_path, img)
    if (i%250 == 0):
        print(f"{i} images processed!")
    features.append(extract_features(img))
f.close()

np.save(sys.argv[3], features)