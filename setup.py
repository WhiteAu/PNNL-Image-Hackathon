from distutils.core import setup

setup(
    name='workingtitle',
    version='0.0.1',
    author='PNWP Hackathon Group'
    author_email=''
    packages=['workingtitle'],
    scripts=[],
    url='https://github.com/WhiteAu/PNNL-Image-Hackathon',
    license='LICENSE.txt',
    description='Work from the September 2017 PNWP Hackathon',
    long_description=open('README.txt').read(),
    install_requires=[
        "pylidc >= 1.8.0"
    ],
)

