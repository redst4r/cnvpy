from setuptools import setup, find_packages

setup(name='cnvpy',
      version=0.1,
      description='A few wrappers around tools to determine CNV in scRNAseq data',
      url='http://github.com/redst4r/cnvpy/',
      author='redst4r',
      maintainer='redst4r',
      maintainer_email='redst4r@web.de',
      license='GNU GPL 3',
      keywords='scrnaseq',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'fastcluster',
          'seaborn',
          'matplotlib',
          'scanpy',
          'tqdm',
          'scikit-learn'
          ],
      zip_safe=False)
