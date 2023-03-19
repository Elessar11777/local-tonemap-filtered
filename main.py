import multiprocessing as mp
from load_images import load_images
import numpy as np
from local_crf import local_crf
from irradiance import compute_irradiance
from tonemap import local_tonemap
import os
import cv2
import gmic

def process_data_folders(root_dir, image_ext, compute_crf, kwargs):
    folder_paths = []
    for data_foldername in os.listdir(root_dir):
        data_folder_path = os.path.join(root_dir, data_foldername)
        if os.path.isdir(data_folder_path):
            for foldername in os.listdir(data_folder_path):
                images_folder_path = os.path.join(data_folder_path, foldername + "/")
                folder_paths.append((foldername, images_folder_path, data_foldername))
    with mp.Pool() as pool:
        pool.starmap(run_hdr, [(folder_name, images_folder_path, data, image_ext, compute_crf, kwargs) for folder_name, images_folder_path, data in folder_paths])

def run_hdr(folder_name, images_folder_path, data, image_ext, compute_crf, kwargs):
    if (len(kwargs) > 0):
        lambda_ = kwargs['lambda_']
        num_px = kwargs['num_px']
        gamma_local = kwargs['gamma_local']
        saturation_local = kwargs['saturation_local']

    [images, B] = load_images(images_folder_path, image_ext)

    if (compute_crf):
        [crf_channel, log_irrad_channel, w] = local_crf(images, B, lambda_=lambda_, num_px=num_px)
    else:
        hdr_loc = kwargs['hdr_loc']
        [crf_channel, log_irrad_channel, w] = np.load(hdr_loc)

    irradiance_map = compute_irradiance(crf_channel, w, images, B)

    # compute locally tonemapped image
    local_tonemapped_image = local_tonemap(irradiance_map, saturation=saturation_local, gamma_=gamma_local, numtiles=(36, 36))
    img = local_tonemapped_image

    result_folder_path = os.path.join("./results/", data)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    path = os.path.join(result_folder_path, folder_name + ".png")
    path = path.replace(" ", "&")
    #cv2.imwrite(path, img)

    gmic_image = gmic.GmicImage.from_numpy(img)
    gmic.run(f"mirror y -fx_unsharp 1,10,20,2,0,2,1,1,0,0 output {path}", gmic_image)

if __name__ == "__main__":
    ROOT_DIR = "./images/"
    IMAGE_EXT = "*.bmp"
    COMPUTE_CRF = True

    kwargs = {'lambda_': 50, 'num_px': 150, 'gamma': 1 / 2.2, 'alpha': 0.35, 'hdr_loc': ROOT_DIR + "crf.npy",
              'gamma_local': 1.0, 'saturation_local': 2.5}

    process_data_folders(ROOT_DIR, IMAGE_EXT, COMPUTE_CRF, kwargs)
