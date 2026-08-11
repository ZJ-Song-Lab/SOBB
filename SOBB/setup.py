from setuptools import setup,find_packages
import os
path = os.path.join(os.path.dirname(__file__),"python")

with open(os.path.join(path, "sobb/__init__.py"), "r", encoding='utf8') as fh:
    for line in fh:
        if line.startswith('__version__'):
            version = line.split("'")[1]
            break
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name="sobb",
    version=version,
    author="Jittor Group",
    author_email="jittor@qq.com",
    description="SOBB: An Analytic Deterministic Framework for SAR Ship Detection",
    url="https://github.com/ZJ-Song-Lab/SOBB",
    python_requires='>=3.7',
    packages=find_packages(path),
    package_dir={'': "python"},
    install_requires=[
        "shapely",
        "pyyaml",
        "numpy",
        "tqdm",
        "pillow",
        "astunparse",
        "jittor",
        "tensorboardX",
        "opencv-python",
        "tqdm",
        "terminaltables",
    ],
)