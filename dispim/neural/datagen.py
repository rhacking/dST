import numpy as np
from pymrt.geometry import sphere, cylinder
from scipy.ndimage import rotate, gaussian_filter, zoom
from scipy.signal import fftconvolve


def gen_test_image(shape=512, n_spheres=2048, n_cylinders=1024, sphere_radius_min=0.15, sphere_radius_mult=1.9,
                   cylinder_radius_min=0.5, cylinder_radius_mult=0.6, cylinder_height_min=20,
                   cylinder_height_mult=80) -> np.ndarray:
    img = np.zeros(shape if type(shape) == tuple else (shape,) * 3)

    for i in range(n_spheres):
        radius = min(np.random.exponential(1) * sphere_radius_mult + sphere_radius_min, img.shape[2] // 2 - 10)
        s = sphere(np.ceil(radius).astype(np.int) * 2 + 1, radius)

        x, y, z = (np.random.randint(0, img.shape[0] - s.shape[0] - 1),
                   np.random.randint(0, img.shape[1] - s.shape[1] - 1),
                   np.random.randint(0, img.shape[2] - s.shape[2] - 1))

        w, h, l = s.shape

        img[x:x + w, y:y + h, z:z + l] += s

    for i in range(n_cylinders):
        radius = min(np.random.exponential(1) * cylinder_radius_mult + cylinder_radius_min, img.shape[2] // 2 - 10)
        height = min(np.random.random() * cylinder_height_mult + cylinder_height_min, img.shape[2] - 10)
        s = cylinder(max(np.ceil(height).astype(np.int), np.ceil(radius).astype(np.int) * 2 + 1), height,
                     radius).astype(np.float)

        for j in range(2):
            axes = [0, 1, 2]
            del axes[j]
            s = rotate(s, np.random.rand() * 180, axes=axes, order=2)

        s = s[:img.shape[0] - 2, :img.shape[1] - 2, :img.shape[2] - 2]

        x, y, z = (np.random.randint(0, img.shape[0] - s.shape[0] - 1),
                   np.random.randint(0, img.shape[1] - s.shape[1] - 1),
                   np.random.randint(0, img.shape[2] - s.shape[2] - 1))

        w, h, l = s.shape

        img[x:x + w, y:y + h, z:z + l] += s

    return img


def degrade_image(img: np.ndarray, psf_size=17, psf_lateral_sigma=1, psf_axial_sigma=4, subsample_factor=0.75,
                  use_poisson=True, gauss_noise_sigma=0., rotated=True) -> np.ndarray:
    # Image a
    img_degr_a = img.copy()
    if rotated:
        img_degr_a = rotate(img_degr_a, 45, (1, 2))

    if psf_lateral_sigma != 0 and psf_axial_sigma != 0:
        psf = np.zeros((psf_size,) * 3)
        psf[psf_size // 2, psf_size // 2, psf_size // 2] = 1
        psf = gaussian_filter(psf, (psf_lateral_sigma,) * 2 + (psf_axial_sigma,))

        img_degr_a = fftconvolve(img_degr_a, psf, mode='same')

    if use_poisson:
        img_degr_a = np.random.poisson(np.maximum(0, img_degr_a)).astype(np.float32)

    if gauss_noise_sigma > 0:
        noise = np.random.normal(0, gauss_noise_sigma, size=img_degr_a.shape).astype(np.float32)
        img_degr_a = np.maximum(0, img_degr_a + noise)

    if subsample_factor != 1:
        img_degr_a = zoom(zoom(img_degr_a, (1, 1, subsample_factor), order=0), (1, 1, 1 / subsample_factor), order=3)

    if rotated:
        img_degr_a = rotate(img_degr_a, -45, (1, 2))
        img_degr_a = img_degr_a[:, int(img.shape[1]//2):-int(img.shape[1]//2), int(img.shape[2]//2):-int(img.shape[2]//2)]

    # Image b
    img_degr_b = img.copy()
    if rotated:
        img_degr_b = rotate(img_degr_b, 45, (1, 2))

    if psf_lateral_sigma != 0 and psf_axial_sigma != 0:
        psf = np.zeros((psf_size,) * 3)
        psf[psf_size // 2, psf_size // 2, psf_size // 2] = 1
        psf = gaussian_filter(psf, (psf_lateral_sigma, psf_axial_sigma, psf_lateral_sigma))

        img_degr_b = fftconvolve(img_degr_b, psf, mode='same')

    if use_poisson:
        img_degr_b = np.random.poisson(np.maximum(0, img_degr_b)).astype(np.float32)

    if gauss_noise_sigma > 0:
        noise = np.random.normal(0, gauss_noise_sigma, size=img_degr_b.shape).astype(np.float32)
        img_degr_b = np.maximum(0, img_degr_b + noise)

    if subsample_factor != 1:
        img_degr_b = zoom(zoom(img_degr_b, (1, subsample_factor, 1), order=0), (1, 1 / subsample_factor, 1), order=3)

    if rotated:
        img_degr_b = rotate(img_degr_b, -45, (1, 2), )
        img_degr_b = img_degr_b[:, int(img.shape[1]//2):-int(img.shape[1]//2), int(img.shape[2]//2):-int(img.shape[2]//2)]

    img_degr_a = img_degr_a[:img_degr_b.shape[0], :img_degr_b.shape[1], :img_degr_b.shape[2]]
    img_degr_b = img_degr_b[:img_degr_a.shape[0], :img_degr_a.shape[1], :img_degr_a.shape[2]]

    return np.stack([img_degr_a, img_degr_b], axis=3)


def gen_training_data(n: int, use_noise: bool, use_psf: bool, use_subsampling: bool, shape: int = 144):
    from progressbar import ProgressBar
    import warnings

    images = []

    print('Generating images...')
    with ProgressBar(max_value=n) as bar:
        for i in bar(range(n)):
            images.append(gen_test_image(shape, n_spheres=240, n_cylinders=100))

    images_degr = []

    print('Degrading images...')
    with ProgressBar(max_value=len(images)) as bar:
        for i in bar(range(len(images))):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='scipy')
                img = images[i]
                img_degr = degrade_image(img, psf_lateral_sigma=np.random.random() * 1.2 if use_psf else 0,
                                         psf_axial_sigma=np.random.random() * 5 if use_psf else 0,
                                         use_poisson=np.random.random() < 0.5 if use_noise else False,
                                         subsample_factor=np.random.random() * 0.5 + 0.5 if use_subsampling else 1.0,
                                         gauss_noise_sigma=np.random.random() * 0.42 if use_noise else 0.0)
                images[i] = images[i][:img_degr.shape[0], :img_degr.shape[1], :img_degr.shape[2]]
                img_degr = img_degr[:images[i].shape[0], :images[i].shape[1], :images[i].shape[2]]
                images_degr.append(img_degr)
                assert images[i].shape[:3] == images_degr[i].shape[:3]
    images = [np.stack([img, np.zeros_like(img)], axis=3) for img in images]

    return images_degr, images

