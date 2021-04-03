import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md") as fh:
    long_description = fh.read()

setuptools.setup(
    name='multiner',
    version="0.0.1",
    description="multilingual named entity recognition with xlm-roberta models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.8.8",
        "Topic :: Named Entity Recognition (multilingual)"
    ],
    install_requires=required,
    author='Uğurcan Özalp',
    author_email='uurcann94@gmail.com'
 )