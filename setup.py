from setuptools import setup, find_packages

readme = open('README.md').read()

VERSION = '0.0.1'

requirements = [
    'onnx'
]

setup(
    name='onnxp',
    version=VERSION,
    author='Zhijian Liu',
    author_email='zhijianliu.cs@gmail.com',
    url='https://github.com/Lyken17/pytorch-OpCounter/',
    description='A simple yet useful profiler for NN models (currently supporting ONNX and PyTorch models).',
    long_description=readme,
    license='MIT',

    packages=find_packages(exclude=('*test*',)),

    zip_safe=True,
    install_requires=requirements,

    classifiers=[
        'Programming Language :: Python :: 3'
    ]
)
