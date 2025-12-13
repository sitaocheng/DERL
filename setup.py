from setuptools import setup, find_packages

setup(
    name='derl',
    version='0.1.0',
    description='Differentiable Evolutionary Reinforcement Learning',
    author='DERL Team',
    author_email='',
    url='https://github.com/sitaocheng/DERL',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'torch>=2.6.0',
        'gym>=0.18.0',
        'matplotlib>=3.3.0',
        'tensorboard>=2.4.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
