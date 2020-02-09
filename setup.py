from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='DSAE_PBHL',
    version='2.0.0',
    description='Package of the autoencoder (AE), sparse autoencoder (SAE), SAE with parametric bias in hidden layer (SAE with PBHL)',
    long_description=readme,
    author='Ryo Ozaki',
    author_email='ryo.ozaki@em.ci.ritsumei.ac.jp',
    url='https://github.com/RyoOzaki/DSAE-PBHL-v2',
    license=license,
    install_requires=['numpy', 'tensorflow>=2.1.0'],
    packages=['DSAE_PBHL']
)
