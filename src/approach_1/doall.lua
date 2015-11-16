
require 'torch'

print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('### DeepDrone  - We need to go deeper ###')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 43, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 3, 'number of threads')
-- data:
cmd:option('-size', 'full', 'how many samples do we load: xsmall | small | full ')
-- model:
cmd:option('-model', 'inception', 'type of model to construct: linear | mlp | convnet | inception | and many more')
-- loss:
cmd:option('-loss', 'simple', 'type of loss function to minimize: nll | mse | margin | simple')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS | ADADELTA | ADAGRAD (recommended)')
cmd:option('-learningRate', 5e-2, 'learning rate at t=0')
cmd:option('-batchSize', 5, 'mini-batch size (1 = pure stochastic)')
cmd:option('-batchForward', true, 'Forward input in mini batches or one at a time')
cmd:option('-weightDecay', 0.0005, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'float', 'type: double | float | cuda')
cmd:option('-visualize', false, 'visualize weights of the network (true | false)')
cmd:option('-dof', 1, 'degrees of freedom; 1: only x coordinates, 2: x, y; 3:x, y, z.')
cmd:option('-saveModel', true, 'Save model after each iteration')
cmd:option('-baseDir', '/scratch/vstrobel/locfiles/draug/', 'Base dir for images and targets')
cmd:option('-regression', true, 'Regression or classification')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile 'data.lua'
dofile 'model.lua'
dofile 'loss.lua'
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------
print '==> training!'

-- Train and test forever (or until one stops the script)
while true do
   train()
   test()
end
