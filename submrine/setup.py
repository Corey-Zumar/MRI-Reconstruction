from setuptools import setup
import os

with open('LICENSE.txt', "r") as f:
    license = f.read()

with open("VERSION.txt", "r") as fp:
    version = fp.read().strip()

setup(
    name='submrine',
    version=version,
    description='',
    maintainer='Corey Zumar, Alex Kot',
    maintainer_email='czumar@berkeley.edu, akot@berkeley.edu',
    license=license,
    packages=["submrine", "submrine.utils", "submrine.train", "submrine.eval"],
    package_data={'submrine': ['*.txt']},
    keywords=['submrine', 'mri', 'reconstruction', 'deep learning'],
    install_requires=[
        'numpy>=1.13.1', 'nibabel>=2.2.0', 'matplotlib>=2.0.2', 'keras>=2.0.6',
        'scikit-image>=0.13.1'
    ],
    entry_points={
        "console_scripts": [
            'submrine-train = submrine.train.train_net:main',
            'submrine-eval = submrine.eval.eval_net:main'
        ]
    })
