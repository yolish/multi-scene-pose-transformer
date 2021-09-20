# A toy script to demonstrate plotting of attention maps
import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2

filename = 'ms-transformer-activation-maps-example.pickle'
with open(filename, 'rb') as handle:
    data = pickle.load(handle)

img = data['img'].squeeze().data.cpu().numpy().transpose(1, 2, 0)
img = 255 * (img - img.min()) / (img.max() - img.min())
img = np.uint8(img)

# plt.figure()
# plt.imshow(img)
# plt.show()

activations = data['acts']
for k,v in activations.items():
	print (k, v.size())

act_img = activations['layers.4._merger.3'].squeeze().data.cpu().numpy()
act_img = np.mean(act_img.reshape(-1, 14, 14), axis=0)
act_img = 255 * (act_img - act_img.min()) / (act_img.max() - act_img.min())
act_img = np.uint8(act_img)

disp_act_img = cv2.resize(act_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title('Input Image')
ax[1].imshow(img)
ax[1].imshow(disp_act_img, alpha=0.3, cmap='jet')
ax[1].axis('off')
ax[1].set_title('Attention')
plt.show()