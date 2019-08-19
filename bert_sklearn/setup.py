from setuptools import find_packages, setup

setup(
    name="bert_sklearn",
    version="0.3.0",
    author="charles_nainan",
    author_email="charles.j.9n@gmail.com",
    description="A sklearn wrapper for Bert",
    keywords='scikit sklearn BERT sklearn NLP deep learning google',
    license='Apache',
    url="https://github.com/charles9n/bert-sklearn",
    packages=find_packages(exclude=['test', 'scripts', 'examples']),
    install_requires=['torch>=0.4.1',
                       'scikit-learn',
                       'numpy',
                       'pandas',
                       'boto3',
                       'requests',
                       'tqdm'],
    python_requires='>=3.5.0',
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
