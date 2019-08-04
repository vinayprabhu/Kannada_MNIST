Morpho-MNIST: Original MNIST morphometrics
==========================================

This dataset consists of the following files:

- [train|t10k]-morpho.csv: morphometrics table, with columns:
    - 'index': index of the corresponding digit (for convenience, although rows
      are written in order)
    - 'area' (px²), 'length' (px), 'thickness' (px), 'slant' (rad), 'width'
      (px), 'height' (px): calculated morphometrics

As in the original MNIST, 'train' and 't10k' refer to the 60,000 training and
10,000 test samples, respectively. The images and labels can be downloaded from
Yann LeCun's website: http://yann.lecun.com/exdb/mnist.

More information, code, and the download links for the other Morpho-MNIST
datasets can be found in our GitHub repository:
https://github.com/dccastro/Morpho-MNIST

Please consider citing the accompanying paper if using this data in your
publications:

Castro, Daniel C., Tan, Jeremy, Kainz, Bernhard, Konukoglu, Ender, and Glocker,
  Ben (2018). Morpho-MNIST: Quantitative Assessment and Diagnostics for
  Representation Learning. arXiv preprint arXiv:1809.10780.
  https://arxiv.org/abs/1809.10780

Contact: Daniel Coelho de Castro <dcdecastro@gmail.com>
