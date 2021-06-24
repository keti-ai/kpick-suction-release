import setuptools
from Cython.Build import cythonize
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

# def get_ext_paths(root_dir, exclude_files):
#     """get filepaths for compilation"""
#     paths = []
#
#     for root, dirs, files in os.walk(root_dir):
#         for filename in files:
#             if os.path.splitext(filename)[1] != '.py':
#                 continue
#
#             file_path = os.path.join(root, filename)
#             if file_path in exclude_files:
#                 continue
#
#             paths.append(file_path)
#     return paths

setuptools.setup(
    name='ketisdk', 
    version='1.0.0',
    author="Keti starteam",
    author_email="bmtrungvp@gmail.com",
    description="Keti SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keti-ai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    license='MIT', 
    keywords = ['VIOSN', 'ROBOT', 'CALIBRATION', 'WORKCELL'],
    install_requires=[            
          'numpy',
        'opencv-python',
         'scipy',
         'matplotlib'
     ],
)	
