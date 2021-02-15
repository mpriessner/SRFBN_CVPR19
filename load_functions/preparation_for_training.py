import numpy as np
from skimage import io
import math
from imageio import imwrite
import imageio
from skimage.transform import rescale, resize, downscale_local_mean
from tempfile import mkstemp
from os import fdopen, remove
from shutil import move, copymode
from shutil import copyfile



def correct_channels(img):
  '''For 2D + T (with or without RGB) a artificial z channel gets created'''
  if img.shape[-1] ==3:
    use_RGB = True
  else:
    use_RGB = False
  if len(img.shape) ==4 and use_RGB:
    t, x, y, c = img.shape
    zeros = np.zeros((t,1,y,x,c))
    zeros[:,0,:,:,:] = img
    img = zeros
  elif len(img.shape) ==3 and not use_RGB:
    t, x, y = img.shape
    zeros = np.zeros((t,1,y,x))
    zeros[:,0,:,:] = img
    img = zeros
  return img, use_RGB
    

def load_img(path):
  # correct images to a 4D or 3D or 5D dataset
  img = io.imread(path)
  img, use_RGB = correct_channels(img)
  return img, use_RGB

def convert_gray3RGB(img):
    print("Image dimensions are:{}".format(img.shape))  # (x, y)
    RGB = np.zeros((img.shape[1], img.shape[0], 3)) # array uses different order (y, x)
    RGB[:,:,0] = img
    RGB[:,:,1] = img
    RGB[:,:,2] = img
    return RGB


def save_HR_LR(img, size_ratio, path, idx, scale, save_HR_path, save_LR_path):
		# HR_img = misc.imresize(img, size, interp='bicubic')
    x,y = img.shape
    x_new = math.ceil(size_ratio*x)
    y_new = math.ceil(size_ratio*y)
    HR_img = resize(img, (y_new, x_new), anti_aliasing=True)
    HR_img = modcrop(HR_img, scale)
    DS_ratio = (1/scale)
    x_downscale_img = resize(HR_img, (math.ceil(y_new*DS_ratio),math.ceil(x_new*DS_ratio)),anti_aliasing=True)

    img_path = path.split('/')[-1].split('.')[0]  + '-ds-' + str(idx) + '.png'
    x_downscale_img_path = path.split('/')[-1].split('.')[0] + '-ds-' + str(idx) + '.png'

    io.imsave(save_HR_path + '/' + img_path, HR_img)
    io.imsave(save_LR_path + '/' + x_downscale_img_path, x_downscale_img)


def modcrop(image, scale):
    if len(image.shape) == 3:
        h, w, _   = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def muiltiporcessing_func(scale, HR_size, save_HR_path, save_LR_path, path):
    print('Processing-----{}/0800'.format(path.split('/')[-1].split('.')[0]))
    img = imageio.imread(path)
    idx = 0
    for size_ratio in HR_size:
        save_HR_LR(img, size_ratio, path, idx, scale, save_HR_path, save_LR_path)
        idx += 1    


from tempfile import mkstemp
from os import fdopen, remove
from shutil import move, copymode
from shutil import copyfile

def change_train_GMFN_file(scale_factor, HR_x_scale_folder, LR_x_scale_folder, HR_val_folder, LR_val_folder, num_epochs, pretrain, pretrained_path):
    # """This function changes the resolution value in the file: Vimeo7_dataset.py"""
    file_path_2 = "/content/SRFBN_CVPR19/options/train/train_GMFN.json"
    fh_2, abs_path_2 = mkstemp()
    with fdopen(fh_2,'w') as new_file:
        with open(file_path_2) as old_file:
            for counter, line in enumerate(old_file):
                if counter ==5:
                   new_file.write(f'    "scale": {scale_factor},\n')
                elif counter ==15:
                   new_file.write(f'            "dataroot_HR": "{HR_x_scale_folder}",\n')          
                elif counter ==16:
                   new_file.write(f'            "dataroot_LR": "{LR_x_scale_folder}",\n')
                elif counter ==27:
                   new_file.write(f'            "dataroot_HR": "{HR_val_folder}",\n')          
                elif counter ==28:
                   new_file.write(f'            "dataroot_LR": "{LR_val_folder}",\n')
                elif counter ==53:
                   new_file.write(f'        "num_epochs": {num_epochs},\n')
                elif counter ==58:
                   if pretrain == "null":
                      new_file.write(f'        "pretrain": null,\n')
                   else:
                      new_file.write(f'        "pretrain": {pretrain},\n')
                elif counter ==59:
                    pretrain_new = "/content/SRFBN_CVPR19/Experiments/GMFN_in3f64_template/epochs/Best_ckp_new.pth"
                    copyfile(pretrained_path, pretrain_new)
                    new_file.write(f'        "pretrained_path": "{pretrain_new}",\n')
                else:
                  new_file.write(line)
    copymode(file_path_2, abs_path_2)
    #Remove original file
    remove(file_path_2)
    #Move new file
    move(abs_path_2, file_path_2) 
