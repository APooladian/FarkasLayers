# FarkasLayers
Public repository for Farkas Layers, as described in [Farkas Layers: Don't shift the data, fix the geometry](https://arxiv.org/abs/1910.02840).

Abstract: Successfully training deep neural networks often requires either batch normalization, appropriate weight initialization, both of which come with their own challenges. We propose an alternative, geometrically motivated method for training. Using elementary results from linear programming, we introduce Farkas layers: a method that ensures at least one neuron is active at a given layer. Focusing on residual networks with ReLU activation, we empirically demonstrate a significant improvement in training capacity in the absence of batch normalization or methods of initialization across a broad range of network sizes on benchmark datasets.

## Using this repo:
This is coded using PyTorch 1.0. The "models" folder contains implementations of standard Residual Networks and FarkasNets, with the corresponding Farkas Blocks (Linear, Convolutional, Residual, etc). Simply modify "run.sh" in the experiment_template folder, and train a FarkasNet. In run.sh, you can tinker with the parameters (i.e. weight initialization, having the last block be zero, etc) to mimick our examples from the paper. To change the aggregation function, you can go into "models/farkas.py" and change the relevant "sum" calls to "mean" calls.

#### To Do
1) Give option for mean or aggregation function in argparser

### Citation
If you find the Farkas Layers useful in your scientific work, please cite as
```
@article{pooladian2019farkaslayers,
  title={Farkas Layers: Don't shift the data, fix the geometry},
  author={Pooladian, Aram-Alexandre and Finlay, Chris and Oberman, Adam M.},
  journal={arXiv preprint arXiv:1910.02840},
  year={2019},
  url={http://arxiv.org/abs/1910.02840},
  archivePrefix={arXiv},
  eprint={1910.02840},
}
```
