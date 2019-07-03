import shutil
from typing import Dict, List, Union, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from csbdeep.data import RawData, create_patches
from csbdeep.models import Config, CARE
from scipy.ndimage import rotate
from scipy.stats import uniform, norm, bernoulli

from .datagen import gen_test_image, degrade_image, gen_training_data


def gen_care_single_model(name: str) -> CARE:
    """
    Generate a single channel CARE model or retrieve the one that already exists under the name specified

    :param name: The name of the model
    :return: The CARE model
    """
    try:
        model = CARE(None, name)
    except FileNotFoundError:
        config = Config('xyzc', n_channel_in=1, n_channel_out=1)
        model = CARE(config, name)

    return model


def gen_care_dual_model(name: str, batch_size: int = 16, **kwargs):
    """
    Generate a dual channel CARE model or retrieve the one that already exists under the name specified

    :param name: The name of the model
    :param batch_size: The training batch size to use (only used if the model doesn't exist yet)
    :param kwargs: Parameters to pass to the model constructor (only used if the model doesn't exist yet
    :return: The CARE model
    """
    try:
        model = CARE(None, name)
    except FileNotFoundError:
        config = Config('xyzc', n_channel_in=2, n_channel_out=1, train_batch_size=batch_size, **kwargs)
        model = CARE(config, name)

    return model


def gen_raw_data(n: int, **kwargs):
    """
    Generate a csbdeep RawData object with random images generated by :func:`my text
    <dispim.neural.datagen.gen_training_data>`

    :param n:
    :param use_noise:
    :param use_psf:
    :param use_subsampling:
    :param shape:
    :return:
    """
    images_degr, images, psf_as, psf_bs = gen_training_data(n, **kwargs)
    return RawData.from_arrays(images_degr, images, axes='XYZC'), psf_as, psf_bs


def split_chunks(data: List[np.ndarray], n_chunks: int, psf_as: List[np.ndarray], psf_bs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray]:
    """
    Extract chunks from RawData and normalize them

    :param data: The data to extract the chunks form
    :param n_chunks: The number of chunks to extract
    :param psf_as: The list of PSF corresponding to the first channel of data
    :param psf_bs: The list of PSF corresponding to the second channel of data
    :return: (X chunks, Y chunks, axes, psf_as, psf_bs)
    """

    def _normalize(patches_x, patches_y, x, y, mask, channel):
        return patches_x, patches_y
    X, Y, axes = create_patches(data, (64, 64, 64, 2), n_patches_per_image=n_chunks, patch_filter=None, shuffle=False)
    return X, Y, axes, np.repeat(psf_as, n_chunks, axis=0), np.repeat(psf_bs, n_chunks, axis=0),


def train_care_generated_data(model: CARE, epochs: int, X: np.ndarray, Y: np.ndarray, X_val: np.ndarray,
                              Y_val: np.ndarray, steps_per_epoch: int = 400) -> None:
    """
    Train a CARE model on a dataset

    :param model: The CARE model to train
    :param epochs: The number of epochs to train for
    :param X: The training data input
    :param Y: The training data expected output
    :param X_val: The validation data input
    :param Y_val: The validation data expected output
    """
    rearr = lambda arr: np.moveaxis(arr, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
    rearry = lambda arr: np.moveaxis(arr, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])[:, :, :, :, :1]
    X = rearr(X)
    X_val = rearr(X_val)
    Y = rearry(Y)
    Y_val = rearry(Y_val)

    model.train(X, Y, validation_data=(X_val, Y_val), epochs=epochs, steps_per_epoch=steps_per_epoch)


def test_model(model: CARE, x_test: np.ndarray, y_test: np.ndarray, batch_size: int = 4) -> Tuple[Any]:
    """
    Test the performance of a CARE model on test data

    :param model: The CARE model to test
    :param x_test: The test data input
    :param y_test: The test data expected output
    :param batch_size: The batch size to use when evaluating the model
    :return: The evaluation metrics as defined by the model (loss, followed by metrics)
    """
    rearr = lambda arr: np.moveaxis(arr, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
    rearry = lambda arr: np.moveaxis(arr, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])[:, :, :, :, :1]
    X = rearr(x_test)
    Y = rearry(y_test)

    return model.keras_model.evaluate(X, Y, batch_size=batch_size)[1]

DEFAULT_DIST = {
              'n_spheres_dist': norm(loc=120, scale=5),
              'n_cubes_dist': norm(loc=0, scale=0),
              'n_cylinders_dist': norm(loc=18, scale=5),
              'psf_lateral_sigma_dist': uniform(loc=0, scale=1),
              'psf_axial_sigma_dist': uniform(loc=0.5, scale=4),
              'use_poisson_dist': bernoulli(0.75),
              'subsample_factor_dist': uniform(loc=0.5, scale=0.5),
              'gauss_noise_sigma_dist': uniform(loc=0, scale=0.42)
}

def gen_model_eval_data(params: Dict[str, Union[List[float], np.ndarray]], shape: int,
                   use_noise: bool, use_psf: bool, use_subsampling: bool, samples_per_param: int = 18, dists=DEFAULT_DIST):
    """
    Analyze how the performance of a CARE model is affected by different parameters of degradation and image content

    :param model: The model to evaluate
    :param params: The parameters to test as a dictionary. The keys must be parameters to test and the values must
                    be values to test
    :param shape: The size of the test images generated to evaluate the performance of the model
    :param use_noise: Whether to use add noise to generated images
    :param use_psf: Whether to convolve the generated images with some PSF
    :param use_subsampling: Whether to downsample the generate images
    :return: A dictionary containing the results
    """
    from tqdm.auto import tqdm
    import warnings
    import tifffile
    import os
    import shutil
    import time
    import matplotlib.pyplot as plt

    results = {}

    if os.path.exists('eval_data'):
        shutil.rmtree('eval_data')

    os.mkdir('eval_data')

    time.sleep(5)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='scipy')
        warnings.filterwarnings('ignore', module='scikits.fitting')
        for param in tqdm(params, desc='Param'):
            os.mkdir(f'eval_data/{param}')
            Y, X, psfas, psfbs = _gen_param_influence_data(param, params[param], shape, use_noise, use_psf,
                                                      use_subsampling, samples_per_param=samples_per_param, dists=dists)
            # print(param, params[param], shape, use_noise, use_psf,
            #                                           use_subsampling, samples_per_param, dists)
            # plt.imshow(np.sum(Y[0][0][:, :, :], axis=0))
            # plt.show()
            # plt.imshow(np.sum(X[0][0][:, :, :, 0], axis=0))
            # plt.show()
            # plt.hist(X[0][0].ravel(), bins=50)
            # plt.show()
            # print(X[0][0].shape, X[0][0].dtype, X[0][0].min(), X[0][0].max(), np.any(np.isnan(X[0][0])), X[0][0].flags)
            for v_i in range(len(X)):
                os.mkdir(f'eval_data/{param}/{params[param][v_i]}')
                os.mkdir(f'eval_data/{param}/{params[param][v_i]}/Y')
                os.mkdir(f'eval_data/{param}/{params[param][v_i]}/X')
                os.mkdir(f'eval_data/{param}/{params[param][v_i]}/psfas')
                os.mkdir(f'eval_data/{param}/{params[param][v_i]}/psfbs')
                # os.mkdir(f'eval_data/{param}/{params[param][v_i]}/X_psf')

                for s in range(len(X[v_i])):
                    tifffile.imsave(f'eval_data/{param}/{params[param][v_i]}/Y/{s}.tif', Y[v_i][s])
                    tifffile.imsave(f'eval_data/{param}/{params[param][v_i]}/X/{s}.tif', X[v_i][s])
                    tifffile.imsave(f'eval_data/{param}/{params[param][v_i]}/psfas/{s}.tif', psfas[v_i][s])
                    tifffile.imsave(f'eval_data/{param}/{params[param][v_i]}/psfbs/{s}.tif', psfbs[v_i][s])
                    # tifffile.imsave(f'eval_data/{param}/{params[param][v_i]}/X_psf/{s}.tif', X_psf[v_i][s])


def plot_model_test_samples(model: CARE, x_test: np.ndarray, y_test: np.ndarray, n: int) -> None:
    """
    Plot examples of the performance of a dual-channel input CARE model

    :param model: The CARE model to test
    :param x_test: The test data input
    :param y_test: The test data expected output
    :param n: The number of examples to plot
    """
    import matplotlib.pyplot as plt

    rearr = lambda arr: np.moveaxis(arr, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
    rearry = lambda arr: np.moveaxis(arr, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])[:, :, :, :, :1]
    X = rearr(x_test)
    if y_test.ndim < 5:
        Y = rearry(y_test[:, np.newaxis, :, :, :])
    else:
        Y = rearry(y_test)

    for i in range(n):
        id = np.random.randint(len(X))
        pred = model.keras_model.predict(X[id:id + 1, :, :, :, :])[0, :, :, :, 0]

        for ax in range(3):
            plt.figure(figsize=(15, 4))
            plt.suptitle(f'Axis {ax}')
            plt.subplot(141)
            plt.grid()
            plt.title('A')
            plt.imshow(np.sum(X[id, :, :, :, 0], axis=ax))

            plt.subplot(142)
            plt.grid()
            plt.title('B')
            plt.imshow(np.sum(X[id, :, :, :, 1], axis=ax))

            plt.subplot(143)
            plt.title('Prediction')
            plt.imshow(np.sum(pred, axis=ax))
            plt.grid()

            plt.subplot(144)
            plt.grid()
            plt.title('Ground truth')
            plt.imshow(np.sum(Y[id, :, :, :, 0], axis=ax))

            plt.show()

        print()


def _gen_param_influence_data(param: str, values: Union[List[float], np.ndarray], shape: int,
                              use_noise: bool, use_psf: bool, use_subsampling: bool, dists, samples_per_param: int = 18):
    from scikits.fitting import GaussianFit
    from tqdm.auto import tqdm

    # FIXME: NORMALIZATION!

    psf_outer = int(shape // 4)
    psf_shape = shape - psf_outer * 2

    if param not in ['n_spheres', 'n_cylinders']:
        img = gen_test_image(shape=shape, n_spheres=int(dists['n_spheres_dist'].rvs()), n_cylinders=int(dists['n_cylinders_dist'].rvs()), n_cubes=int(dists['n_cubes_dist'].rvs()))
        img_psf = img.copy()
        img_psf[psf_outer:-psf_outer, psf_outer:-psf_outer, psf_outer:-psf_outer] = 0
        img_psf[int(shape // 2), int(shape // 2), int(shape // 2)] = 1

    # df = pd.DataFrame(columns=[param, 'mse', 'sigma_x', 'sigma_y', 'sigma_z'], dtype=np.float32)
    images = []
    images_degr = []
    psfas = []
    psfbs = []
    # images_degr_psf = []
    for v in tqdm(values, desc='Value', leave=False):
        param_samples = []
        param_samples_degr = []
        param_samples_degr_psf = []
        psfa_samples = []
        psfb_samples = []
        for i in tqdm(range(samples_per_param), leave=False, desc='Image'):
            success = False
            if param in ['n_spheres', 'n_cylinders', 'n_cubes']:
                params = {'n_spheres': int(dists['n_spheres_dist'].rvs()), 'n_cylinders': int(dists['n_cylinders_dist'].rvs()), 'n_cubes': int(dists['n_cubes_dist'].rvs())}
                params[param] = v
                img = gen_test_image(shape=shape, **params)
                img_psf = img.copy()
                img_psf[psf_outer:-psf_outer, psf_outer:-psf_outer, psf_outer:-psf_outer] = 0
                img_psf[int(shape // 2), int(shape // 2), int(shape // 2)] = 1
            while not success:
                try:
                    # params = {'psf_lateral_sigma': np.random.random() * 1.2 if use_psf else 0,
                    #           'psf_axial_sigma': np.random.random() * 5 if use_psf else 0,
                    #           'use_poisson': np.random.random() < 0.5 if use_noise else False,
                    #           'subsample_factor': np.random.random() * 0.5 + 0.5 if use_subsampling else 1.0,
                    #           'gauss_noise_sigma': np.random.random() * 0.42 if use_noise else 0.0}
                    params = {param.replace('_dist', ''): param_dist.rvs() for param, param_dist in dists.items()}
                    del params['n_spheres']
                    del params['n_cylinders']
                    del params['n_cubes']
                    if param in params.keys():
                        params[param] = v
                    elif param not in ['n_spheres', 'n_cylinders', 'n_cubes']:
                        raise ValueError(f'Unknown parameter: {param}')

                    param_samples.append(img)
                    degr = degrade_image(img, **params)
                    param_samples_degr.append(degr[0][:shape, :shape, :shape])
                    psfa_samples.append(degr[1])
                    psfb_samples.append(degr[2])
                    # min_shape = np.min([degr.shape, img.shape], axis=0)
                    # mse = np.mean(np.square(
                    #     img[:min_shape[0], :min_shape[1], :min_shape[2]] - model.keras_model.predict(
                    #         degr[np.newaxis, :min_shape[0], :min_shape[1], :min_shape[2]])[0, :, :, :, 0]))

                    # degr_psf = degrade_image(img_psf, **params)
                    # param_samples_degr_psf.append(degr_psf[0][:shape, :shape, :shape])
                    # min_shape = np.min([degr_psf.shape, img_psf.shape], axis=0)
                    # pred_psf = model.keras_model.predict(
                    #     degr_psf[np.newaxis, :min_shape[0], :min_shape[1], :min_shape[2]])[0, :, :, :, 0]
                    # pred_psf = rotate(pred_psf, 45, (1, 2))
                    #
                    # gauss = GaussianFit((int(psf_shape // 2), int(psf_shape // 2), int(psf_shape // 2)), (1, 1, 1), 1)
                    # x = np.array(list(np.ndindex(psf_shape, psf_shape, psf_shape)))
                    # y = [pred_psf[psf_outer:-psf_outer, psf_outer:-psf_outer, psf_outer:-psf_outer][
                    #          coord[0], coord[1], coord[2]] for coord in x]
                    # gauss.fit(x.T, y)
                    #
                    # df.loc[len(df)] = [v, mse, gauss.std[0], gauss.std[1], gauss.std[2]]
                    success = True
                except tf.errors.InvalidArgumentError:
                    pass
        images.append(param_samples)
        images_degr.append(param_samples_degr)
        psfas.append(psfa_samples)
        psfbs.append(psfb_samples)
        # images_degr_psf.append(param_samples_degr_psf)

    return images, images_degr, psfas, psfbs#, images_degr_psf

