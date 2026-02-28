import math
from os.path import dirname, join

import numpy as np
import scipy.io
import scipy.ndimage
import scipy.special
import scipy.linalg
from PIL import Image

from pathlib import Path

# replacement for scipy.misc.imresize using Pillow
def imresize_np(img, scale, resample=Image.BICUBIC):
    """
    Resize a 2D numpy array (float32) by a scale factor using Pillow.

    img: 2D numpy array
    scale: float, e.g. 0.5 for half-size
    """
    # img shape is (h, w)
    h, w = img.shape
    new_size = (int(w * scale), int(h * scale))  # (width, height)

    # ensure float32, Pillow mode "F" = 32-bit floating point pixels
    pil_img = Image.fromarray(img.astype(np.float32), mode="F")
    pil_resized = pil_img.resize(new_size, resample=resample)

    return np.array(pil_resized, dtype=np.float32)


gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)


def aggd_features(imdata):
    # flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = 0.0
    right_mean_sqrt = 0.0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt / right_mean_sqrt
    else:
        gamma_hat = np.inf
    # solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (
        ((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1))
        / math.pow(math.pow(gamma_hat, 2) + 1, 2)
    )

    # solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm) ** 2)
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # mean parameter
    N = (br - bl) * (gam2 / gam1)  # *aggdratio
    return alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt


def ggd_features(imdata):
    nr_gam = 1 / prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq / E ** 2
    pos = np.argmin(np.abs(nr_gam - rho))
    return gamma_range[pos], sigma_sq


def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return H_img, V_img, D1_img, D2_img


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    total = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        total += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= total
    return weights


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode="constant"):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype("float32")
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return (image - mu_image) / (var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array(
        [
            alpha_m,
            (bl + br) / 2.0,
            alpha1,
            N1,
            bl1,
            br1,  # (V)
            alpha2,
            N2,
            bl2,
            br2,  # (H)
            alpha3,
            N3,
            bl3,
            bl3,  # (D1)
            alpha4,
            N4,
            bl4,
            bl4,  # (D2)
        ]
    )


def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)


def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)


def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j : j + patch_size, i : i + patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features


def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = h % patch_size
    woffset = w % patch_size

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)
    # replaced scipy.misc.imresize with Pillow-based function
    img2 = imresize_np(img, 0.5)

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size / 2)

    feats = np.hstack((feats_lvl1, feats_lvl2))

    return feats


def niqe(inputImgData):
    patch_size = 96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(join(module_path, "data", "niqe_image_params.mat"))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]

    M, N = inputImgData.shape

    assert (
        M > (patch_size * 2 + 1)
    ), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert (
        N > (patch_size * 2 + 1)
    ), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = (pop_cov + sample_cov) / 2.0
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score


def print_niqe_scores(folder_path):
    folder = Path(folder_path)

    # collect png + jpg + jpeg (also covers uppercase extensions)
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        image_paths.extend(folder.glob(ext))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"No images found in: {folder}")
        return

    scores = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("LA")
            gray = np.array(img)[:, :, 0]
            score = niqe(gray)
            scores.append(score)
            print(f"NIQE of {p.name} is: {score:0.3f}")
        except Exception as e:
            print(f"Failed on {p.name}: {e}")

    if scores:
        avg_score = float(np.mean(scores))
        print(f"\nAverage NIQE for folder '{folder.name}' over {len(scores)} images is: {avg_score:0.3f}")
    else:
        print("\nNo NIQE scores were computed (all images failed).")



if __name__ == "__main__":
    dry1 = np.array(Image.open("./test_imgs/dry1.png").convert("LA"))[:, :, 0]  
    dry2 = np.array(Image.open("./test_imgs/dry2.png").convert("LA"))[:, :, 0]  
    dry22 = np.array(Image.open("./test_imgs/dry22.png").convert("LA"))[:, :, 0]  


    print("NIQE of dry_gen1 image is: %0.3f" % niqe(dry1))
    print("NIQE of dry_gen2 image is: %0.3f" % niqe(dry2))
    print("NIQE of dry_original2 image is: %0.3f" % niqe(dry22))

    con1 = np.array(Image.open("./test_imgs/con1.png").convert("LA"))[:, :, 0]  
    con2 = np.array( Image.open("./test_imgs/con2.png").convert("LA"))[:, :, 0]  
    con3 = np.array( Image.open("./test_imgs/con3.png").convert("LA"))[:, :, 0] 
    con11 = np.array(Image.open("./test_imgs/con11.png").convert("LA"))[:, :, 0]  
    con22 = np.array( Image.open("./test_imgs/con22.png").convert("LA"))[:, :, 0]  
    con33 = np.array( Image.open("./test_imgs/con33.png").convert("LA"))[:, :, 0]  

    print("NIQE of con_gen1 image is: %0.3f" % niqe(con1))
    print("NIQE of con_gen2 image is: %0.3f" % niqe(con2))
    print("NIQE of con_gen3 image is: %0.3f" % niqe(con3))
    print("NIQE of con_original1 image is: %0.3f" % niqe(con11))
    print("NIQE of con_original2 image is: %0.3f" % niqe(con22))
    print("NIQE of con_original3 image is: %0.3f" % niqe(con33))

    new_dry_1 = np.array(Image.open("./test_imgs/ver_271225/Dry/image_001.png").convert("LA"))[:, :, 0]
    print("NIQE of Dry_test_1 image is: %0.3f" % niqe(new_dry_1))

    print_niqe_scores(r"./test_imgs/ver_271225/Dry")
    print("--------------------------------------------------------------")
    print_niqe_scores(r"./test_imgs/Dried")



  
