# Deepdrone - Convolutional Networks for Localization

Work in progress for developing a robust camera pose estimation system
in [Torch](http://torch.ch/docs/getting-started.html#_). It needs a
couple of additional packages, you can install them with:

- `luarocks install dp`
- `luarocks install csvigo`
- ...

## Usage

1. Generate different views from an image using
[draug](https://github.com/tudelft/draug).

2. Run the script (`doall.lua`) and specify additional parameters. The
additional flags that can be found in doall.lua. For example:

   `th -i doall.lua -baseDir /home/volker/draug/genimgs/ -model
   inception -type cuda -batchSize 5 -saveModel -optimization
   ADADELTA`

This command would call the script and

- Set the base dir to /home/volker/draug/genimgs/
- Set the model to inception ([GoogLeNet](http://arxiv.org/abs/1409.4842) variant)
- Specify that CUDA should be used
- Set the batch size to 5 (that is 5 inputs are fed in to the model simualtaneously)
- Does NOT save the model
- Set the optimization to [ADADELTA](http://arxiv.org/abs/1212.5701)


## Further Explanations

The general structure and many functions are from the following torch
tutorial:
[https://github.com/torch/tutorials/tree/master/2_supervised](https://github.com/torch/tutorials/tree/master/2_supervised). The
inception architecture is from
[https://github.com/nicholas-leonard/dp/blob/master/examples/deepinception.lua](https://github.com/nicholas-leonard/dp/blob/master/examples/deepinception.lua).

`doall.lua` processes the following files:

- 'data.lua'

Reads in and rescales images (from `draug/genimgs`), sets targets from
`targets.csv`

- 'model.lua' (requires 'deepinception.lua')

This files includes many different network architectures. The best performing one is `inception`.

- 'loss.lua'

Specifies the loss function used for learning. For regression this is
the MSECriterion. For classification it is the Kullback–Leibler
divergence.

- 'train.lua'

Trains the model and adds the predictions to a confusion matrix

- 'test.lua'


## TODO

There's still a lot to do, this repository is in a very early stage.

- Can only predict x coordinates (1 DoF)

- Always assumes that there are 350 different x positions, that is,
  that the draug output is between 0 and 350 (easy fix).

- ...