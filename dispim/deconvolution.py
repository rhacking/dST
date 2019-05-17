import logging

import numpy as np
import progressbar

from dispim import Volume, metrack

logger = logging.getLogger(__name__)


def deconvolve_single(vol_a: Volume, n: int, psf_a: np.ndarray) -> Volume:
    """
    Perform Richardson-Lucy deconvolution on a single volume using the specified PSF on the CPU

    :param vol_a: The first volume
    :param n: The number of Richardson-Lucy iterations
    :param psf_a: The PSF for the first volume
    """
    # from astropy.convolution import convolve_fft
    from functools import partial
    from scipy.signal import fftconvolve
    view_a = vol_a.astype(np.float)

    psf_a = psf_a.astype(np.float) / np.sum(psf_a).astype(np.float)
    psf_Ai = psf_a[::-1, ::-1, ::-1]

    estimate = view_a.copy()

    convolve = partial(fftconvolve, mode='same')

    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            estimate = estimate * convolve(view_a / (convolve(estimate, psf_a) + 1), psf_Ai)

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    # TODO: Rescaling might be unwanted
    e_min, e_max = np.percentile(estimate, [0.05, 99.95])
    estimate = ((np.clip(estimate, e_min, e_max) - e_min) / (e_max - e_min) * (2 ** 16 - 1)).astype(np.uint16)

    return Volume(estimate, inverted=False, spacing=vol_a.spacing, is_skewed=False)


def deconvolve_single_gpu(vol_a: Volume, n: int, psf_a: np.ndarray) -> Volume:
    """
    Perform joint Richardson-Lucy deconvolution on two volumes using two specified PSFs on the GPU

    :param vol_a: The first volume
    :param n: The number of Richardson-Lucy iterations
    :param psf_a: The PSF for the first volume
    :return: The fused RL deconvolution
    """
    from functools import partial
    from dispim.metrics import DECONV_MSE_DELTA
    import arrayfire as af

    print(vol_a.shape)

    view_a = vol_a.astype(np.float)

    psf_a = psf_a.astype(np.float) / np.sum(psf_a).astype(np.float)
    psf_Ai = psf_a[::-1, ::-1, ::-1]

    view_a = af.cast(af.from_ndarray(np.array(view_a)), af.Dtype.f32)

    psf_a = af.cast(af.from_ndarray(psf_a), af.Dtype.f32)
    psf_Ai = af.cast(af.from_ndarray(psf_Ai), af.Dtype.f32)

    estimate = view_a

    convolve = partial(af.fft_convolve3)

    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            if metrack.is_tracked(DECONV_MSE_DELTA):
                prev = estimate
            estimate = estimate * convolve(view_a / (convolve(estimate, psf_a) + 1), psf_Ai)

            if metrack.is_tracked(DECONV_MSE_DELTA):
                metrack.append_metric(DECONV_MSE_DELTA, (_, float(np.mean((prev - estimate) ** 2))))

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    logger.debug(f'Deconved min: {np.min(estimate)}, max: {np.max(estimate)}, has nan: {np.any(np.isnan(estimate))}')

    result = estimate.to_ndarray()
    del estimate

    return Volume(result.astype(np.uint16), inverted=False, spacing=vol_a.spacing, is_skewed=False)


def deconvolve(vol_a: Volume, vol_b: Volume, n: int, psf_a: np.ndarray, psf_b: np.ndarray) -> Volume:
    """
    Perform joint Richardson-Lucy deconvolution on two volumes using two specified PSFs on the CPU

    :param vol_a: The first volume
    :param vol_b: The second volume
    :param n: The number of Richardson-Lucy iterations
    :param psf_a: The PSF for the first volume
    :param psf_b: The PSF for the second volume
    """
    # from astropy.convolution import convolve_fft
    from functools import partial
    from scipy.signal import fftconvolve
    view_a, view_b = vol_a.astype(np.float), vol_b.astype(np.float)

    psf_a = psf_a.astype(np.float) / np.sum(psf_a).astype(np.float)
    psf_b = psf_b.astype(np.float) / np.sum(psf_b).astype(np.float)
    psf_Ai = psf_a[::-1, ::-1, ::-1]
    psf_Bi = psf_b[::-1, ::-1, ::-1]

    estimate = (view_a + view_b) / 2

    convolve = partial(fftconvolve, mode='same')

    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            estimate = estimate * convolve(view_a / (convolve(estimate, psf_a) + 1e-6), psf_Ai)
            estimate = estimate * convolve(view_b / (convolve(estimate, psf_b) + 1e-6), psf_Bi)

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    # TODO: Rescaling might be unwanted
    e_min, e_max = np.percentile(estimate, [0.05, 99.95])
    estimate = ((np.clip(estimate, e_min, e_max) - e_min) / (e_max - e_min) * (2 ** 16 - 1)).astype(np.uint16)

    return Volume(estimate, inverted=False, spacing=vol_a.spacing, is_skewed=False)


def deconvolve_gpu_chunked(vol_a: Volume, vol_b: Volume, n: int, psf_a: np.ndarray, psf_b: np.ndarray, nchunks: int,
                           blind: bool = False) -> Volume:
    """
    Perform joint Richardson-Lucy deconvolution on two volumes using two specified PSFs on the GPU in chunks along the
    z-axis

    :param vol_a: The first volume
    :param vol_b: The second volume
    :param n: The number of Richardson-Lucy iterations
    :param psf_a: The PSF for the first volume
    :param psf_b: The PSF for the second volume
    :param blind: Whether to perform blind RL deconvolution using the given PSFs as initial estimates
    :param nchunks: The number of chunks to subdivide the volume into
    :return: The fused RL deconvolution
    """
    import arrayfire as af
    result = np.zeros(vol_a.shape, np.float32)
    chunk_size = vol_a.shape[2] // nchunks
    for i in range(nchunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < nchunks - 1 else vol_a.shape[2]
        lpad = int(psf_a.shape[2] * 4)
        rpad = int(psf_a.shape[2] * 4)

        start_exp = max(0, start - lpad)
        end_exp = min(vol_a.shape[2], end + rpad)

        with metrack.Context(f'Chunk {i}'):
            if not blind:
                chunk_est = deconvolve_gpu(Volume(vol_a[:, :, start_exp:end_exp], False, (1, 1, 1)),
                                           Volume(vol_b[:, :, start_exp:end_exp], False, (1, 1, 1)), n, psf_a,
                                           psf_b)
            else:
                chunk_est = deconvolve_gpu_blind(Volume(vol_a[:, :, start_exp:end_exp], False, (1, 1, 1)),
                                                 Volume(vol_b[:, :, start_exp:end_exp], False, (1, 1, 1)), n, 5,
                                                 psf_a, psf_b)

        af.device_gc()

        if end != end_exp:
            result[:, :, start:end] = chunk_est[:, :, start - start_exp:end - end_exp]
        else:
            result[:, :, start:end] = chunk_est[:, :, start - start_exp:]

    # FIXME: Proper outlier clipping!
    e_min, e_max = np.percentile(result, [0.002, 99.998])
    result = ((np.clip(result, e_min, e_max) - e_min) / (e_max - e_min) * (2 ** 16 - 1)).astype(np.uint16)

    return Volume(result, inverted=False, spacing=(1, 1, 1))


def deconvolve_gpu(vol_a: Volume, vol_b: Volume, n: int, psf_a: np.ndarray, psf_b: np.ndarray) -> Volume:
    """
    Perform joint Richardson-Lucy deconvolution on two volumes using two specified PSFs on the GPU

    :param vol_a: The first volume
    :param vol_b: The second volume
    :param n: The number of Richardson-Lucy iterations
    :param psf_a: The PSF for the first volume
    :param psf_b: The PSF for the second volume
    :return: The fused RL deconvolution
    """
    from functools import partial
    from dispim.metrics import DECONV_MSE_DELTA
    import arrayfire as af

    print(vol_a.shape, vol_b.shape)

    view_a, view_b = vol_a.astype(np.float), vol_b.astype(np.float)

    psf_a = psf_a.astype(np.float) / np.sum(psf_a).astype(np.float)
    psf_b = psf_b.astype(np.float) / np.sum(psf_b).astype(np.float)
    psf_Ai = psf_a[::-1, ::-1, ::-1]
    psf_Bi = psf_b[::-1, ::-1, ::-1]

    view_a = af.cast(af.from_ndarray(np.array(view_a)), af.Dtype.f32)
    view_b = af.cast(af.from_ndarray(np.array(view_b)), af.Dtype.f32)

    psf_a = af.cast(af.from_ndarray(psf_a), af.Dtype.f32)
    psf_b = af.cast(af.from_ndarray(psf_b), af.Dtype.f32)
    psf_Ai = af.cast(af.from_ndarray(psf_Ai), af.Dtype.f32)
    psf_Bi = af.cast(af.from_ndarray(psf_Bi), af.Dtype.f32)

    estimate = (view_a + view_b) / 2

    convolve = partial(af.fft_convolve3)

    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            if metrack.is_tracked(DECONV_MSE_DELTA):
                prev = estimate
            estimate_A = estimate * convolve(view_a / (convolve(estimate, psf_a) + 1), psf_Ai)
            estimate = estimate_A * convolve(view_b / (convolve(estimate, psf_b) + 1), psf_Bi)

            if metrack.is_tracked(DECONV_MSE_DELTA):
                metrack.append_metric(DECONV_MSE_DELTA, (_, float(np.mean((prev - estimate) ** 2))))

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    logger.debug(f'Deconved min: {np.min(estimate)}, max: {np.max(estimate)}, has nan: {np.any(np.isnan(estimate))}')

    return Volume(estimate.to_ndarray(), inverted=False, spacing=vol_a.spacing, is_skewed=False)


def deconvolve_gpu_blind(vol_a: Volume, vol_b: Volume, n: int, m: int, psf_a: np.ndarray, psf_b: np.ndarray) -> Volume:
    """
    Perform blind joint Richardson-Lucy deconvolution on two volumes using two specified estimates of the PSF on the
    GPU

    :param vol_a: The first volume
    :param vol_b: The second volume
    :param n: The number of Richardson-Lucy iterations
    :param m: The number of sub-iterations per RL iteration
    :param psf_a: The initial PSF estimate for the first volume
    :param psf_b: The initial PSF estimate for the second volume
    :return: The fused RL deconvolution
    """
    from functools import partial
    import arrayfire as af
    view_a, view_b = vol_a.astype(np.float), vol_b.astype(np.float)

    psf_a = psf_a.astype(np.float) / np.sum(psf_a).astype(np.float)
    psf_b = psf_b.astype(np.float) / np.sum(psf_b).astype(np.float)
    padding = tuple(
        (int(s // 2 - psf_a.shape[i]), int((s - s // 2) - psf_a.shape[i])) for i, s in enumerate(view_a.shape))
    psf_a = np.pad(psf_a,
                   tuple(((s - psf_a.shape[i]) // 2, (s - psf_a.shape[i]) - (s - psf_a.shape[i]) // 2) for i, s in
                         enumerate(view_a.shape)), 'constant')
    psf_b = np.pad(psf_b,
                   tuple(((s - psf_b.shape[i]) // 2, (s - psf_b.shape[i]) - (s - psf_b.shape[i]) // 2) for i, s in
                         enumerate(view_b.shape)), 'constant')

    view_a = af.cast(af.from_ndarray(view_a), af.Dtype.u16)
    view_b = af.cast(af.from_ndarray(view_b), af.Dtype.u16)

    psf_a = af.cast(af.from_ndarray(psf_a), af.Dtype.f32)
    psf_b = af.cast(af.from_ndarray(psf_b), af.Dtype.f32)

    estimate = (view_a + view_b) / 2

    convolve = partial(af.fft_convolve3)

    lamb = 0.002

    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            for j in range(m):
                psf_a = psf_a * convolve(view_a / (convolve(psf_a, estimate) + 1e-1), estimate[::-1, ::-1, ::-1])
            for j in range(m):
                estimate = estimate * convolve(view_a / (convolve(estimate, psf_a) + 10), psf_a[::-1, ::-1, ::-1])
            for j in range(m):
                psf_b = psf_b * convolve(view_b / (convolve(psf_b, estimate) + 1e-1), estimate[::-1, ::-1, ::-1])
            for j in range(m):
                estimate = estimate * convolve(view_b / (convolve(estimate, psf_b) + 10), psf_b[::-1, ::-1, ::-1])

    del psf_a, psf_b, view_a, view_b

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    return Volume(estimate.to_ndarray(), inverted=False, spacing=vol_a.spacing, is_skewed=False)


def extract_psf(vol: Volume, min_size: int = 5, max_size: int = 30, psf_half_width: int = 10) -> np.ndarray:
    """
    Attempt to extract the PSF from a volume by looking at small objects that are representative of the volume's PSF

    :param vol: The volume to extract the PSF from
    :param min_size: The minimum area of objects to consider
    :param max_size: The maximum area of objects to consider
    :param psf_half_width: The half-width of the PSF in all axes
    :return: The estimated PSF, shape = (psf_half_width*2+1, ) * 3
    """
    from skimage.measure import label, regionprops
    from dispim.util import extract_3d, threshold_otsu
    from dispim.metrics import PSF_SIGMA_XY, PSF_SIGMA_Z
    data = vol
    thr = threshold_otsu(data[:, :, ])
    data_bin = data > thr

    points = np.array([np.array(r.centroid, dtype=np.int) for r in regionprops(label(data_bin))
                       if min_size <= r.area <= max_size])

    logger.debug(f'Found {len(points)} objects')

    # points = np.random.choice(points, size=min(len(points), 12000), replace=False)
    points = points[np.random.choice(len(points), min(len(points), 12000), replace=False), :]

    blob_images = []
    for point in points:
        blob_images.append(extract_3d(data, point, psf_half_width))

        if metrack.is_tracked(PSF_SIGMA_XY) or metrack.is_tracked(PSF_SIGMA_Z):
            height, center_x, center_y, width_x, width_y, rotation = fitgaussian(blob_images[-1][psf_half_width, :, :])
            scale = vol.shape[0]
            if width_x > width_y:
                metrack.append_metric(PSF_SIGMA_Z, (None, width_x * scale))
                metrack.append_metric(PSF_SIGMA_XY, (None, width_y * scale))
            else:
                metrack.append_metric(PSF_SIGMA_Z, (None, width_y * scale))
                metrack.append_metric(PSF_SIGMA_XY, (None, width_x * scale))

    median_blob = np.median(blob_images, axis=0)

    logger.debug(f'PSF mean: {median_blob.mean()}, median: {np.median(median_blob)}, min: {median_blob.min()}, max: {median_blob.max()}')

    return median_blob


def fuse(vol_a: Volume, vol_b: Volume) -> Volume:
    """
    Fuse two volumes, without RL deconvolution

    :param vol_a: The first volume
    :param vol_b: The second volume
    :return: The fused volume
    """
    return ((vol_a + vol_b) / 2).astype(np.uint16)


def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)
    center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x, y):
        xp = x * np.cos(rotation) - y * np.sin(rotation)
        yp = x * np.sin(rotation) + y * np.cos(rotation)
        g = height * np.exp(
            -(((center_x - xp) / width_x) ** 2 +
              ((center_y - yp) / width_y) ** 2) / 2.)
        return g

    return rotgauss


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0.0


def fitgaussian(data):
    import scipy.optimize
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = scipy.optimize.leastsq(errorfunction, params)
    return p
