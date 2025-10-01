"""
setup.py - Package Installation Configuration
==============================================
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text()
else:
    long_description = "Floor Plan Panel/Joist Optimization System"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'Pillow>=9.0.0',
        'matplotlib>=3.4.0',
        'pytesseract>=0.3.8',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'jinja2>=3.0.0',
        'seaborn>=0.11.0'
    ]

setup(
    name='floorplan-optimizer',
    version='1.0.0',
    author='Floor Plan Optimization Team',
    author_email='support@floorplanoptimizer.com',
    description='A comprehensive system for optimizing panel/joist placement in floor plans',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/floorplan-optimizer',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Manufacturing',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'black>=21.6b0',
            'flake8>=3.9.0',
            'mypy>=0.910',
            'coverage>=5.5',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
        ],
        'dxf': [
            'ezdxf>=0.17.0',
        ],
        'ml': [
            'scikit-learn>=1.0.0',
            'networkx>=2.6',
        ]
    },
    entry_points={
        'console_scripts': [
            'floorplan-optimize=main:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.json', '*.yaml', '*.yml', '*.txt'],
        'data': ['patterns/*.json', 'models/*.pkl'],
    },
    zip_safe=False,
)
