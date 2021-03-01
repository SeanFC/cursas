import setuptools

__version__ = "0.0.1"

name = 'cursas'
version = __version__
release = version 

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requires = [ line.strip() for line in fh ]

setuptools.setup(
    name=name,
    version=release,
    author="Sean F. Cleator",
    author_email="seancleator@hotmail.co.uk",
    description="Visualise data from parkrun",
    license="GNU GPLv3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_namespace_packages(include=['cursas']),#, "docs", "tests"]),
    python_requires='>=3.9',
    install_requires=(requires),
    zip_safe=False, # This removes some installation errors
    )
