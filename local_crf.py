import numpy as np
from crf import crf_solve

def local_crf(images, B, lambda_=50, num_px=150):
    num_images = len(images)
    Zmin = 0
    Zmax = 255

    # image parameters
    H, W, C = images[0].shape

    # optmization parameters
    px_idx = np.random.choice(H * W, (num_px,), replace=False)

    # define pixel intensity weighting function w
    w = np.concatenate((np.arange(128) - Zmin, Zmax - np.arange(128, 256)))

    # compute Z matrix
    Z = np.empty((num_px, num_images))
    crf_channel = []
    log_irrad_channel = []
    for ch in range(C):
        for j, image in enumerate(images):
            flat_image = image[:, :, ch].flatten()
            Z[:, j] = flat_image[px_idx]

        # get crf and irradiance for each color channel
        [crf, log_irrad] = crf_solve(Z.astype('int32'), B, lambda_, w, Zmin, Zmax)
        crf_channel.append(crf)
        log_irrad_channel.append(log_irrad)

    return [crf_channel, log_irrad_channel, w]