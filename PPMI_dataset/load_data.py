import nibabel as nib
import os
import numpy as np
import hickle as hkl
import csv
import matplotlib.pyplot as plt


# save ppmi dataset as images in ./data folder
# unified size of images: (144, 240, 256)

def save_images_ppmi(base_path):
    dir_list = os.listdir(base_path)
    cnt = 0
    cnt2 = 0
    dim1 = 144
    dim2 = 240
    for dir_name in dir_list:
        patient_id = int(dir_name)
        # using T1-anatomical folder
        full_path = base_path + dir_name + '/T1-anatomical/'
        dir_list_times = os.listdir(full_path)
        dir_list_times = np.sort(dir_list_times)
        selected_dir = dir_list_times[0]
        full_path = os.path.join(full_path, selected_dir)
        full_path = os.path.join(full_path, os.listdir(full_path)[0])
        file_path = os.path.join(full_path, os.listdir(full_path)[0])

        # 2 crashed files ignored
        ignore_list = ['/home/leslie/PPMI/3851/T1-anatomical/2012-06-29_08_31_11.0/S171235/PPMI_3851_MR_T1-anatomical_Br_20140129161419188_S171235_I411120.nii',
                       '/home/leslie/PPMI/50319/T1-anatomical/2014-11-24_12_56_30.0/S243878/PPMI_50319_MR_T1-anatomical_Br_20150304174950072_S243878_I476196.nii']

        # loop over all files
        if file_path not in ignore_list:
            cnt2 += 1
            file_data = nib.load(file_path)
            image = file_data.get_fdata()

            # check order of dimension
            if image.shape != (176, 240, 256):
                print('shape:', image.shape)

                # reform all case of with wrong order of dimensions
                if image.shape[-1] != 256:
                    image = np.moveaxis(image, [0, 2], [2, 0])
                    print('new shape:', image.shape)
                cnt += 1
            tmp1 = int((image.shape[0] - dim1) / 2)
            tmp2 = int((image.shape[1] - dim2) / 2)
            hkl.dump(image[tmp1: tmp1 + dim1, tmp2: tmp2 + dim2, :], 'data/' + str(patient_id) + '.hkl', 'w')
    print('# of deformed shape: ', cnt)
    print('# total:', cnt2)


# get a list of patient ids based on file names on ./data directory
def get_patient_ids(base_path):
    list_dir = os.listdir(base_path)
    patient_ids = []
    for file in list_dir:
        if file[-3:] == 'hkl':
            patient_ids.append(int(file[:-4]))
    return patient_ids


# load labels of patients
def load_ids_labels(base_path):
    file_name = 'Baseline_Data_Summary.csv'
    full_path = os.path.join(base_path, file_name)
    ids = get_patient_ids(base_path)
    labels = np.zeros((len(ids), ))
    with open(full_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        cnt = 0
        for row in csv_reader:
            # look for labels whose images are available in ./data/ directory
            if (cnt > 0) and (int(row[0]) in ids):
                labels[ids.index(int(row[0]))] = (1 if row[1] == 'HC' else 2)  # 2-class classification
            cnt += 1
    ids = np.asarray(ids)
    labels = np.asarray(labels)
    return remove_patients_without_labels(ids, labels)  # remove ids without ids


def remove_patients_without_labels(ids, labels):
    # if labels[i] == 0 then that id does not have label
    idx = [i for i in range(labels.shape[0]) if labels[i] > 0]
    ids = ids[idx]
    labels = labels[idx]
    return ids, labels


def plot_image(base_path, idx):
    image = hkl.load(base_path)
    for i in idx:
        plt.imshow(image[i, :, :])
        plt.show()

# patient_id = 3108
# idx = [10, 20, 30, 40, 120, 130, 140]
# path = './data/' + str(patient_id) + '.hkl'
# plot_image(path, idx)
