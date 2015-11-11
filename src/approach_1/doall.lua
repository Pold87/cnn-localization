----------------------------------------------------------------------
-- This tutorial shows how to train different models on the street
-- view house number dataset (SVHN),
-- using multiple optimization techniques (SGD, ASGD, CG), and
-- multiple types of models.
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------
require 'torch'

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Deep Drone Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 43, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 3, 'number of threads')
-- data:
cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
-- model:
cmd:option('-model', 'bnmodel', 'type of model to construct: linear | mlp | convnet | volker')
-- loss:
cmd:option('-loss', 'simple', 'type of loss function to minimize: nll | mse | margin | simple')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 5e-2, 'learning rate at t=0')
cmd:option('-batchSize', 5, 'mini-batch size (1 = pure stochastic)')
cmd:option('-batchForward', true, 'Forward input in batches or in a loop')
cmd:option('-weightDecay', 0.0005, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'float', 'type: double | float | cuda')
cmd:option('-visualize', false, 'visualize weights of the network (true | false)')
cmd:option('-dof', 1, 'degrees of freedom; 1: only x coordinates, 2: x, y; 3:x, y, z.')
cmd:option('-saveModel', true, 'Save model after each iteration')
cmd:option('-baseDir', '/home/pold/Documents/draug/', 'Base dir for images and targets')
cmd:option('-regression', true, 'Base directory for images and targets')
cmd:option('-lecunlcn', false, 'apply Yann LeCun Local Contrast Normalization (recommended)')
cmd:option('-standardize', false, 'apply Standardize preprocessing')
cmd:option('-zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('-manPrepro', true, 'Apply preprocessing from torch supervised tutorials')

cmd:option('-dropout', false, 'use dropout')
cmd:option('-dropoutProb', '{0.2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5}', 'dropout probabilities')

cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
--if opt.type == 'float' then
--  print('==> switching to floats')
--   torch.setdefaulttensortype('torch.FloatTensor')
--elseif opt.type == 'cuda' then
--   print('==> switching to CUDA')
--   require 'cunn'
--   torch.setdefaulttensortype('torch.FloatTensor')
--end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile 'data.lua'
dofile 'deepinception.lua'
--dofile 'loss.lua'
--dofile 'train.lua'
--dofile 'test.lua'

----------------------------------------------------------------------
print '==> training!'

--while true do
--   train()
--   test()
--end

--for i=1, 50 do
--    train()
--    test()
--end
