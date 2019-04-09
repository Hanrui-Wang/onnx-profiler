import os

from setuptools import setup

version_file = 'onnxp/version.py'
exec(open(version_file).read())

long_description = ''
if os.path.exists('README.md'):
    with open('README.md', 'r') as fp:
        long_description = fp.read()

setup(
    name='onnxp',
    version=__version__,
    description='A simple yet useful profiler for NN models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Zhijian Liu',
    author_email='zhijianliu.cs@gmail.com',
    license='MIT',
    url='https://github.com/zhijian-liu/onnx-profiler',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    packages=[
        'onnxp.profile',
        'onnxp.profilers'
    ],
    install_requires=[
        'numpy',
        'onnx'
    ]
)
