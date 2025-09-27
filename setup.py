from setuptools import setup, find_packages

setup(
    name='echostrain',  
    version='0.1.0',
    description='CAMUS echocardiography segmentation and strain dataset processing tools',
    author='Eda Özkaynar',  
    author_email='you@example.com',  
    packages=find_packages(),  # Tüm alt klasörleri dahil eder (içinde __init__.py varsa)
    install_requires=[
        'torch>=1.13',
        'torchvision',
        'numpy',
        'pandas',
        'opencv-python',
        'Pillow',
        'scikit-image',
        'SimpleITK',
        'imageio',
        'tqdm',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
