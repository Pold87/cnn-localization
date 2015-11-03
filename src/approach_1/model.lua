----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- models:
--   + linear
--   + 2-layer neural network (MLP)
--   + convolutional network (ConvNet)
--
-- It's a good idea to run this script with the interactive mode:
-- $ th -i 2_model.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to play with the model.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'nnx'      -- provides MultiSoftMax

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Deep Drone Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end






local SpatialMaxout, parent = torch.class('nn.SpatialMaxout','nn.Module')

function SpatialMaxout:__init()
   parent.__init(self)
   self.poolsize = 2
end

function SpatialMaxout:updateOutput(input)
   assert(input:size(2) % self.poolsize == 0)
   self.output:resize(input:size(1), input:size(2) / self.poolsize, 
      input:size(3), input:size(4))

   jzt.SpatialMaxout_costGrad(input, self.output, self.gradInput, 
      self.gradInput, self.poolsize, 0)
   return self.output
end

function SpatialMaxout:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   
   jzt.SpatialMaxout_costGrad(input, self.output, self.gradInput, 
      gradOutput, self.poolsize, 1)
   return self.gradInput
end




----------------------------------------------------------------------
print '==> define parameters'

-- Number of positions
noutputs = 300

-- input dimensions
nfeats = 3
width = 112
height = 112
ninputs = nfeats * width * height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {112,112,224}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs, noutputs))

elseif opt.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs, nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens, noutputs))

elseif opt.model == 'convnet' then

   if opt.type == 'cuda' then
      -- a typical modern convolution network (conv+relu+pool)
      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 3 : standard 2-layer neural network
      model:add(nn.View(nstates[2] * filtsize * filtsize))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(nstates[2] * filtsize * filtsize, nstates[3]))
      model:add(nn.ReLU())
      model:add(nn.Linear(nstates[3], 2 * noutputs))

   else
      -- a typical convolutional network, with locally-normalized hidden
      -- units, and L2-pooling

      -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
      -- work on this dataset (http://arxiv.org/abs/1204.3968). In particular
      -- the use of LP-pooling (with P=2) has a very positive impact on
      -- generalization. Normalization is not done exactly as proposed in
      -- the paper, and low-level (first layer) features are not fed to
      -- the classifier.

      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

      model:add(nn.Reshape(112 * 53 * 53))
--      model:add(nn.Reshape())
      model:add(nn.Linear(112 * 53 * 42, nstates[3]))
      model:add(nn.Tanh())
      model:add(nn.Linear(nstates[3], opt.dof * total_range))

      model:add(nn.Reshape(opt.dof, total_range))


   end

elseif opt.model == 'l2model' then

   model = nn.Sequential()

   model:add(nn.SpatialConvolution(3, 12, 3, 3, 2, 2, 1, 1))
   model:add(nn.SpatialLPPooling(12, 2 , poolsize, poolsize))
   model:add(nn.ReLU(true))
   model:add(nn.SpatialDropout(0.5))


   model:add(nn.SpatialConvolution(12, 8, 4, 4))
   model:add(nn.SpatialLPPooling(8, 2, poolsize, poolsize))
   model:add(nn.ReLU(true))
   model:add(nn.SpatialDropout(0.5))

   model:add(nn.View(8 * 26 * 26))
   model:add(nn.Linear(8 * 26 * 26, opt.dof * total_range))
   model:add(nn.Reshape(opt.dof, total_range))

elseif opt.model == 'convmodel' then

   model = nn.Sequential()

   model:add(nn.SpatialConvolution(3, 6, 5, 5))
--   model:add(nn.ReLU(true))
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   model:add(nn.SpatialDropout(0.5))


   model:add(nn.SpatialConvolution(6, 16, 5, 5))
--   model:add(nn.ReLU(true))
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   model:add(nn.SpatialDropout(0.5))


   model:add(nn.View(16 * 53 * 53))

   model:add(nn.Linear(16 * 53 * 53, opt.dof * total_range))
   model:add(nn.Reshape(opt.dof, total_range))

elseif opt.model == 'allconv' then

   -- It worked worked on train but not on test set 

   model = nn.Sequential()

   model:add(nn.SpatialConvolution(3, 12, 3, 3))
   model:add(nn.SpatialConvolution(12, 16, 3, 3))
   model:add(nn.SpatialConvolution(16, 18, 3, 3))
   model:add(nn.SpatialConvolution(18, 20, 3, 3, 2, 2))
   model:add(nn.SpatialBatchNormalization(20, nil,nil, false))
   model:add(nn.SpatialConvolution(20, 16, 3, 3, 1, 1))
   model:add(nn.SpatialConvolution(16, 12, 3, 3, 1, 1))
   model:add(nn.SpatialConvolution(12, 12, 3, 3, 2, 2))

   model:add(nn.View(12 * 51 * 51))

   model:add(nn.Linear(12 * 51 * 51, opt.dof * total_range))
   model:add(nn.Reshape(opt.dof, total_range))


elseif opt.model == 'allconvbn' then

   -- THIS IS KILLER
   -- BATCH NORM RULEZ !1!11!!

-- Worked perfectly with qlua
--  doall.lua -batchSize 7 -model allconvbn -optimization ADAGRAD 


   model = nn.Sequential()

   model:add(nn.SpatialConvolution(3, 8, 3, 3))
--   model:add(nn.ReLU())
--   model:add(nn.SpatialDropout(0.2))

   model:add(nn.SpatialConvolution(8, 8, 3, 3))
--   model:add(nn.ReLU())
--   model:add(nn.SpatialDropout(0.1))

   model:add(nn.SpatialConvolution(8, 8, 3, 3))
--   model:add(nn.ReLU())
--   model:add(nn.SpatialDropout(0.2))


   model:add(nn.SpatialConvolution(8, 8, 3, 3, 2, 2, 1, 1))
--   model:add(nn.ReLU())
--   model:add(nn.SpatialDropout(0.1))


   model:add(nn.SpatialConvolution(8, 8, 3, 3))
--   model:add(nn.ReLU())
--   model:add(nn.SpatialDropout(0.1))

   model:add(nn.SpatialConvolution(8, 8, 3, 3))
--   model:add(nn.ReLU())
--   model:add(nn.SpatialDropout(0.1))

   model:add(nn.SpatialConvolution(8, 8, 3, 3))


   model:add(nn.SpatialConvolution(8, 8, 3, 3, 2, 2, 1, 1))
--   model:add(nn.ReLU())
--   model:add(nn.SpatialDropout(0.1))
--   model:add(nn.SpatialDropout(0.5))
--   model:add(nn.SpatialBatchNormalization(20, nil,nil, false))

--   model:add(nn.ReLU())
--   model:add(nn.SpatialDropout(0.2))

-- model:add(nn.SpatialConvolution(14, 14, 3, 3, 2, 2))
--   model:add(nn.ReLU())


   model:add(nn.View(8 * 52 * 52))


   model:add(nn.Linear(8 * 52 * 52, opt.dof * total_range))

--   model:add(nn.Dropout(0.5))
   model:add(nn.Reshape(opt.dof, total_range))


elseif opt.model == 'allconvbn2' then

   -- THIS IS KILLER
   -- BATCH NORM RULEZ !1!11!!

-- Worked perfectly with qlua
--  doall.lua -batchSize 7 -model allconvbn -optimization ADAGRAD 


   model = nn.Sequential()

   model:add(nn.SpatialConvolution(3, 24, 3, 3))
--   model:add(nn.ReLU())
   model:add(nn.SpatialDropout(0.3))
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))


   model:add(nn.SpatialConvolution(24, 12, 3, 3))
--   model:add(nn.ReLU())
   model:add(nn.SpatialDropout(0.3))
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))


   model:add(nn.View(12 * 54 * 54))

   model:add(nn.Linear(12 * 54 * 54, opt.dof * total_range))
   model:add(nn.Reshape(opt.dof, total_range))



elseif opt.model == 'allconvcuda' then

   model = nn.Sequential()

   model:add(nn.SpatialConvolution(3, 12, 3, 3))
   model:add(nn.SpatialConvolution(12, 12, 3, 3))
   model:add(nn.SpatialConvolution(12, 12, 3, 3))
   model:add(nn.SpatialConvolution(12, 12, 3, 3, 2, 2))
   model:add(nn.SpatialConvolution(12, 16, 3, 3, 1, 1))

   model:add(nn.View(16 * 53 * 53))

   model:add(nn.Linear(16 * 53 * 53, opt.dof * total_range))
   model:add(nn.Reshape(opt.dof, total_range))

   model:cl()


elseif opt.model == 'bnmodel' then

   model = nn.Sequential()

   model:add(nn.SpatialConvolution(3, 12, 3, 3, 2, 2, 1, 1))
   model:add(nn.SpatialBatchNormalization(12, nil,nil, false))
   model:add(nn.ReLU(true))
   model:add(nn.SpatialDropout(0.1))


   model:add(nn.SpatialConvolution(12, 8, 4, 4))
   model:add(nn.SpatialBatchNormalization(8, nil, nil, false))
   model:add(nn.ReLU(true))
   model:add(nn.SpatialDropout(0.1))


   model:add(nn.View(8 * 109 * 109))
   model:add(nn.Linear(8 * 109 * 109, opt.dof * total_range))
   model:add(nn.Reshape(opt.dof, total_range))


elseif opt.model == 'volkersimple' then

   model = nn.Sequential()
   model:add(nn.View(112 * 112 * 3))

   model:add(nn.Linear(112 * 112 * 3, opt.dof * total_range))

else

   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
	 print '==> visualizing ConvNet filters'
	 print('Layer 1 filters:')
	 itorch.image(model:get(1).weight)
	 print('Layer 2 filters:')
	 itorch.image(model:get(5).weight)
      else
	 print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end
