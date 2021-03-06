from setuptools import setup

if __name__ == "__main__":

    REQUIRED_PACKAGES = ["numpy", "nibabel", "tqdm"]

    setup(
        name="niftyregpy",
        version="0.0.2",
        author="fyrdahl",
        author_email="alexander.fyrdahl@gmail.com",
        description="Python interface for NiftyReg",
        packages=[
            "niftyregpy",
            "niftyregpy.average",
            "niftyregpy.apps",
            "niftyregpy.reg",
            "niftyregpy.tools",
            "niftyregpy.transform",
            "niftyregpy.utils",
        ],
        url="https://github.com/fyrdahl/niftyregpy",
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3 :: Only",
            "Topic :: Scientific/Engineering :: Medical Science Apps",
        ],
        install_requires=REQUIRED_PACKAGES,
        license="MIT",
        project_urls={"Source": "https://github.com/fyrdahl/niftyregpy"},
        package_dir={"": "src"},
    )
