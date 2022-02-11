import sys
from distutils.core import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit('Python < 3.6 is not supported')

REQUIRED_PACKAGES = ['numpy', 'nibabel']

setup(
    name="niftyregpy",
    version="0.0.1",
    author="fyrdahl",
    author_email="fyrdahl@med.umich.edu",
    description="Python interface for NiftyReg",
    url="https://github.com/fyrdahl/niftyregpy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License",
        "Operating System :: OS X",
        "Operating System :: Linux"
    ],
    ext_modules = [],
    package_dir = {},
    packages = ["niftyregpy","niftyregpy.utils", "niftyregpy.reg", "niftyregpy.tools", "niftyregpy.average", "niftyregpy.apps"],
)
