from setuptools import setup

with open("./README.md") as f:
    long_desc: str = f.read()

if __name__ == "__main__":

    REQUIRED_PACKAGES = ["numpy", "nibabel"]

    setup(
        name="niftyregpy",
        version="0.0.2",
        author="fyrdahl",
        author_email="fyrdahl@med.umich.edu",
        description="Python interface for NiftyReg",
        long_description_content_type="text/markdown",
        long_description=long_desc,
        packages=[
            "niftyregpy",
            "niftyregpy.average",
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
        ],
        install_requires=REQUIRED_PACKAGES,
        license="MIT",
        project_urls={
            "Source": "https://github.com/ZeroIntensity/pointers.py",
            "Documentation": "https://pointerspy.netlify.app/",
        },
        package_dir={"": "src"},
    )
