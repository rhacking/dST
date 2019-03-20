from . import metrack

MUTUAL_INFORMATION_METRIC = metrack.Metric('Mutual Information', 'Iteration', 'MI')
MUTUAL_INFORMATION_GRADIENT_METRIC = metrack.Metric('Mutual Information Gradient', 'Iteration', 'MI gradient')

DECONV_MSE_DELTA = metrack.Metric('RL Deconvolution MSE delta', 'Iteration', 'MSE delta')

PROCESS_TIME = metrack.Metric('Process Time', 'Process', 'Time (minutes)')

PSF_SIGMA_Z = metrack.Metric('PSF Axial Sigma', 'dist', '')
PSF_SIGMA_XY = metrack.Metric('PSF Lateral Sigma', 'dist', '')
