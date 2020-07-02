# Jarvis

The Jarvis package is comprised of tools for data science and deep learning optimized for healthcare analytics, including dedicated support for multidimensional data such as images, videos, waveforms and other raw acquisitions. 

The Jarvis package is modular by design, allowing for independent use of specific submodules and/or custom code as needed. Specifically the framework is divided into the main (core) package as well as an ecosystem of extendable plugins. All plugins are available under the `jarvis-md-*` domain as described below. After installation, any new available plugins can be accessed under the same generic namespace e.g. `from jarvis.* import *`.

# Tutorials

* How to use and customize the Jarvis data `Client()` object: [link](https://colab.research.google.com/github/peterchang77/dl_tutor/blob/master/jarvis/notebooks/client/client-use.ipynb)
* How to create a new Jarvis data `Client()`: link 
