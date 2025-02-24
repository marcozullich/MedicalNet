import numpy as np
import nibabel
from nilearn import plotting
import torch

from matplotlib import pyplot as plt

from datasets.brains18 import BrainS18Dataset
from setting import parse_opts
from model import generate_model


# 1. Visualize the data

nii_file = "data/MRBrainS18/images/1.nii.gz"
img = nibabel.load(nii_file)
img_data = img.get_fdata()
print("Image shape: ", img_data.shape)
print(img.header)

# slice_x = img_data[img_data.shape[0] // 2, :, :]
# slice_y = img_data[:, img_data.shape[1] // 2, :]
# slice_z = img_data[:, :, img_data.shape[2] // 2]

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(slice_x.T, cmap="gray", origin="lower")
# axes[0].set_title("Sagittal Plane")

# axes[1].imshow(slice_y.T, cmap="gray", origin="lower")
# axes[1].set_title("Coronal Plane")

# axes[2].imshow(slice_z.T, cmap="gray", origin="lower")
# axes[2].set_title("Axial Plane")

# plt.show()


# segmentation_file = "data/MRBrainS18/labels/1.nii.gz"
# segments = nibabel.load(segmentation_file)
# segments_data = segments.get_fdata()

# plt.figure(figsize=(8, 8))
# plt.imshow(img_data[:, :, img_data.shape[2] // 2].T, cmap="gray", origin="lower")
# plt.imshow(segments_data[:, :, img_data.shape[2] // 2].T, cmap="jet", alpha=0.5)  # Overlay mask
# plt.title("Axial View with Segmentation Overlay")
# plt.show()

# plotting.view_img(nii_file, bg_img=None) did not work on my hardware

# 2. Load the dataset and pretrained model

settings = parse_opts()
settings.phase = "train"
settings.resume_path = "trails/models/resnet_50_epoch_110_batch_0.pth.tar"
settings.model_depth = 50
settings.no_cuda = True # toggle if cuda

dataset = BrainS18Dataset(settings.data_root, settings.img_list, settings)


settings.phase = "test"
model, _ = generate_model(settings)
state_dict = torch.load(settings.resume_path, map_location="cpu")["state_dict"]
# remove "module." from the keys
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

# 3. Info on input and output

data, label = dataset[0]
print("Data shape: ", data.shape)
print("Label shape: ", label.shape)

out = model(torch.from_numpy(data).unsqueeze(0))
print("Output shape: ", out.shape)



