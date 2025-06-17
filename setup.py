from setuptools import setup, find_packages

setup(
    name='pupil_analysis_package',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'pandas',
        'tqdm',
        'scikit-learn',
        'joblib',
    ],
    author='Anton Krasnoyarov',
    description='A Python library for pupil analysis from video files.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Anton97/Eyes',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)


