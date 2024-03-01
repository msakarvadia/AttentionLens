from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.readlines()

setup(
  name="attention_lens", 
  version="0.0.1",
  install_requires=install_requires,
  packages=find_packages())
