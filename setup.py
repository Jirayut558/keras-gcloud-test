from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='Sentiment classification Keras',
      author='Jirayut Keawchuen',
      author_email='jirayut.keawchuen@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py',
          'pandas',
          'scikit-learn'
      ],
      zip_safe=False)