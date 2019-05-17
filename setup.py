from setuptools import setup

setup(
    name='dST',
    version='0.1',
    packages=['dispim', ],
    entry_points={
        'console_scripts': {
            'dispim = dispim.__main__:main'
        }
    }, install_requires=['numba', 'matplotlib', 'numpy', 'progressbar2', 'scipy', 'dipy', 'tifffile',
                         'scikit-image', 'coloredlogs', 'regex', 'scikit-image']
)