import setuptools
import nltk

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md") as fh:
    long_description = fh.read()

nltk.download("punkt")

setuptools.setup(
    name='multiner',
    version="0.0.1",
    description="multilingual named entity recognition with xlm-roberta models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        'License :: OSI Approved :: Apache-2.0 License ',
        "Programming Language :: Python :: 3.8.8",
        "Topic :: Natural Language Processing :: Named Entity Recognition (multilingual)"
    ],
    install_requires=required,
    author='Uğurcan Özalp',
    author_email='uurcann94@gmail.com'
 )