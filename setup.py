import os
import re
import setuptools

NAME             = "tensor-mosaic"
AUTHOR           = "Blake A. Wilson"
AUTHOR_EMAIL     = "wilso692@gmail.com"
DESCRIPTION      = "Slice Cache and Allocator for torch Tensors"
LICENSE          = "MIT"
KEYWORDS         = "torch"
URL              = "https://github.com/btrainwilson/" + NAME
README           = ".github/README.md"
CLASSIFIERS      = [
  "Development Status :: 1 - Planning",
  
]
INSTALL_REQUIRES = [
  "torch",
  
]
ENTRY_POINTS = {
  
}
SCRIPTS = [
  
]

HERE = os.path.dirname(__file__)

def read(file):
  with open(os.path.join(HERE, file), "r") as fh:
    return fh.read()

VERSION = re.search(
  r'__version__ = [\'"]([^\'"]*)[\'"]',
  read(NAME.replace("-", "_") + "/__init__.py")
).group(1)

LONG_DESCRIPTION = read(README)

if __name__ == "__main__":
  setuptools.setup(
    name=NAME,
    version=VERSION,
    packages=setuptools.find_packages(),
    author=AUTHOR,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    keywords=KEYWORDS,
    url=URL,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    entry_points=ENTRY_POINTS,
    scripts=SCRIPTS,
    include_package_data=True    
  )
