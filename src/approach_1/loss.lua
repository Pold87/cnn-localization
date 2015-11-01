----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- loss functions:
--   + negative-log likelihood, using log-normalized output units (SoftMax)
--   + mean-square error
--   + margin loss (SVM-like)
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions
require 'nnx'      -- provides all sorts of loss functions
--require 'dp'      -- provides all sorts of loss functions

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Deep Drone Loss Function')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-loss', 'multi', 'type of loss function to minimize: nll | mse | margin')
   cmd:text()
   opt = cmd:parse(arg or {})

   -- to enable self-contained execution:
   model = nn.Sequential()
end

-- 10-class problem
noutputs = 300

----------------------------------------------------------------------
print '==> define loss'

if opt.loss == 'margin' then

   -- This loss takes a vector of classes, and the index of
   -- the grountruth class as arguments. It is an SVM-like loss
   -- with a default margin of 1.

   criterion = nn.MultiMarginCriterion()

elseif opt.loss == 'nll' then

   -- This loss requires the outputs of the trainable model to
   -- be properly normalized log-probabilities, which can be
   -- achieved using a softmax function

   model:add(nn.LogSoftMax())

   -- The loss works like the MultiMarginCriterion: it takes
   -- a vector of classes, and the index of the grountruth class
   -- as arguments.

   criterion = nn.ClassNLLCriterion()

elseif opt.loss == 'mse' then

   -- for MSE, we add a tanh, to restrict the model's output
   model:add(nn.Tanh())

   -- The mean-square error is not recommended for classification
   -- tasks, as it typically tries to do too much, by exactly modeling
   -- the 1-of-N distribution. For the sake of showing more examples,
   -- we still provide it here:

   criterion = nn.MSECriterion()
   criterion.sizeAverage = false

   -- Compared to the other losses, the MSE criterion needs a distribution
   -- as a target, instead of an index. Indeed, it is a regression loss!
   -- So we need to transform the entire label vectors:

   if trainData then
      -- convert training labels:
      local trsize = (#trainData.labels)[1]
      local trlabels = torch.Tensor( trsize, noutputs )
      trlabels:fill(-1)
      for i = 1,trsize do
         trlabels[{ i,trainData.labels[i] }] = 1
      end
      trainData.labels = trlabels

      -- convert test labels
      local tesize = (#testData.labels)[1]
      local telabels = torch.Tensor( tesize, noutputs )
      telabels:fill(-1)
      for i = 1,tesize do
         telabels[{ i,testData.labels[i] }] = 1
      end
      testData.labels = telabels
   end

elseif opt.loss == 'multi' then

   if opt.dof == 1 then
      model:add(nn.LogSoftMax())
   else
      model:add(nn.MultiSoftMax())
   end

   logModule = nn.Sequential()
   logModule:add(nn.AddConstant(0.00000001)) -- fixes log(0)=NaN errors
   logModule:add(nn.Log())

   if opt.dof == 1 then
      criterion = nn.DistKLDivCriterion()
   else
      criterion = nn.ModuleCriterion(nn.DistKLDivCriterion(), logModule, nn.Convert())
   end

elseif opt.loss == 'simple' then
   
   criterion = nn.DistKLDivCriterion()

   if opt.type == 'cuda' then
      criterion = criterion:cl()
   end

else

   error('unknown -loss')

end

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)
