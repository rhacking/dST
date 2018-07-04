from setuptools import setup

setup(
    name='DST',
    version='0.1',
    package='dispim',
    entry_points={
        'console_scripts': {
            'dispim = dispim.__main__:main'
        }
    }, install_requires=['numba', 'matplotlib', 'numpy', 'progressbar', 'scipy']
)