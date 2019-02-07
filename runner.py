import numpy as np,sys,os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pydicom

from model import UNet


training_dirs = [
    # './MR_data_batch1/1/T1DUAL',
    './train_ct'
    # './train_mri'
    # './CT_data_batch1/1',
    # './CT_data_batch1/2',
    # './CT_data_batch1/5',
    # './CT_data_batch1/6',
    # './CT_data_batch1/8',
    # './CT_data_batch1/10',
    # './CT_data_batch1/14',
    # './CT_data_batch1/16',
    # './CT_data_batch1/18',
    # './CT_data_batch1/19',
]

testing_dirs = [
    './test'
]

training_data = []
ground_data = []


# Collecting all paths to all data
for traindir in training_dirs:
    for dirName, subdirList, fileList in sorted(os.walk(traindir)):
        for filename in sorted(fileList):
            if ".dcm" in filename.lower():
                        training_data.append(os.path.join(dirName,filename))
for traindir in training_dirs:
    for dirName, subdirList, fileList in sorted(os.walk(traindir)):
        for filename in sorted(fileList):
            if ".png" in filename.lower():
                        ground_data.append(os.path.join(dirName,filename))


# Reading dicom training data
def read_dicoms(paths):
    slices = [pydicom.read_file(s) for s in paths]
    slices.sort(key = lambda x: int(x.InstanceNumber))
        
    return slices


# Converting pixel_array of dicom files into hounsfield units, author: Howard Chen
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    x = np.array(image, dtype=np.int16)
    y = np.expand_dims(x, axis=3)
    return y

# For MRI
def get_pixel_arrays(scans):
    image = np.stack([s.pixel_array for s in scans])
    x = np.array(image, dtype=np.int16)
    y = np.expand_dims(x, axis=3)
    return y

# Reading png data for labeling every pixel
def get_ground_data(paths):
    image = np.stack([plt.imread(s) for s in paths])
    image = image.astype(np.int16)
    x = np.array(image, dtype=np.int16)
    y = np.expand_dims(x, axis=3)
    return y


print('Loading ct dicom training data ...')
cts = read_dicoms(training_data)
imgs = get_pixels_hu(cts)
# imgs_mri = get_pixel_arrays(cts)

print('Loading png ground data ...')
png_imgs = get_ground_data(ground_data)


# ########### Plotting hounsfield unit histogram
# output_path = './'
# id=0
# np.save(output_path + "fullimages_%d.npy" % (id), imgs)

# file_used=output_path+"fullimages_%d.npy" % id
# imgs_to_process = np.load(file_used).astype(np.float64) 

# plt.hist(imgs_to_process.flatten(), bins=50, color='c')
# plt.xlabel("HU")
# plt.ylabel("freq")
# plt.show()

# ########### Plotting dicom file using matplotlib
# first_dicom = pydicom.dcmread(training_data[0])
# plt.imshow(first_dicom.pixel_array, cmap=plt.cm.bone)
# plt.show()

# Clipping CT data [-600, 200]
imgs[imgs > 200] = 200
imgs[imgs < -600] = -600

# Create context inputs (sequences of 3 slices)
print('Get contexts inputs ...')
ct_seqs = []
for idx, ct in enumerate(imgs):
  if (idx != 0) and idx != (len(imgs)-1):
      spatial_context = np.stack((imgs[idx-1], imgs[idx], imgs[idx+1]))
      ct_seqs.append(spatial_context)
ct_seqs_np = np.array(ct_seqs)

# Remove first and last output segmentation mask
cut_png_imgs = np.delete(png_imgs, [0, len(png_imgs)-1], 0) # axis = 0

# Initializing u-net and train the network
fcn = UNet()
fcn.train(ct_seqs_np, cut_png_imgs) ##### TRAIN CT
# fcn.train(imgs_mri, png_imgs, (256,256,1)) ##### TRAIN MRI

# # TESTING
# test_data_path = ['./CT_data_batch1/10/DICOM_anon/i0081,0000b.dcm']
# test2 = read_dicoms(test_data_path)
# test1 = get_pixels_hu(test2)

# hi = fcn.predict(test1)

# # OUTPUT FILE plot
# plt.imshow(np.squeeze(hi[0], axis=2))
# plt.show()

print('OK')

