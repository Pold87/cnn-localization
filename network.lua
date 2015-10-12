require 'nn'
require 'clnn'
require 'optim'
require 'ffmpeg'
require 'image'
require 'csvigo'

-- Read CSV and convert to tensor
csv_file = csvigo.load("data.csv")
gps_x = torch.Tensor(csv_file.gps_x)

dspath = '/home/pold/Documents/torchstuff/video.avi'
source = ffmpeg.Video{path=dspath,
                      width=224 / 2, height=224 / 2, 
--                      width=256 / 2, height=256 / 2, 
                      encoding='png', 
                      fps=30, 
--                      length=10, 
                      delete=false, 
                      load=false}

rawFrame = source:forward()

-- input video params:
ivch = rawFrame:size(1) -- channels
ivhe = rawFrame:size(2) -- height
ivwi = rawFrame:size(3) -- width

trainDir = '/home/pold/Documents/torchstuff/train/'
trainImaNumber = 2240 / 2
-- trainImaNumber = 1000

trainset = {
   data = torch.Tensor(trainImaNumber, ivch, ivhe, ivwi),
   label = torch.DoubleTensor(trainImaNumber, 1),
   size = function() return trainImaNumber end
}

testset = {
   data = torch.Tensor(trainImaNumber, ivch, ivhe, ivwi),
   label = torch.DoubleTensor(trainImaNumber, 1),
   size = function() return trainImaNumber end
}



 -- shuffle dataset: get shuffled indices in this variable:
local train_shuffle = torch.randperm(trainImaNumber) -- train shuffle

-- Split the data into train and test images

-- load train data: 2416 images
-- for i = 0, trainImaNumber do
for i = 0, trainImaNumber * 2 - 1 do

  -- get next frame:
   img = source:forward()--:clone()

  i_prime = math.floor(i / 2)

  if i % 2 == 0 then
  -- Decide if it goes to train or test data
     trainset.data[train_shuffle[i_prime + 1]] = img--:clone()
     trainset.label[train_shuffle[i_prime + 1]] = torch.DoubleTensor({gps_x[i + 1]})

  else
     testset.data[train_shuffle[i_prime + 1]] = img--:clone()
     testset.label[train_shuffle[i_prime + 1]] = torch.DoubleTensor({gps_x[i + 1]})
  end

end



-- Split the data into train and test images

-- load train data: 2416 images
-- for i = 0, trainImaNumber do
--for i = 0, trainImaNumber * 2 - 1 do
--
--  -- get next frame:
--   img = source:forward()--:clone()
--
--  i_prime = math.floor(i / 2)
--
--  if i % 2 == 0 then
--  -- Decide if it goes to train or test data
--     trainset.data[i_prime + 1] = img--:clone()
--     trainset.label[i_prime + 1] = torch.DoubleTensor({gps_x[i + 1]})
--
--  else
--     testset.data[i_prime + 1] = img--:clone()
--     testset.label[i_prime + 1] = torch.DoubleTensor({gps_x[i + 1]})
--  end
--
--end
--
-- Begin new file

-- trainset = torch.load('train.t7')
-- testset = torch.load('test.t7')

-- trainset.label = trainset.label:cl()
-- trainset.data = trainset.data:cl()

-- ignore setmetatable for now, it is a feature beyond the scope of this tutorial. It sets the index operator.
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

-- trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future


for i = 1, 3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


-- normalize target

mean_target = {} -- store the mean, to normalize the test set in the future
stdv_target  = {} -- store the standard-deviation for the future

mean_target = trainset.label:mean()
trainset.label:add(-mean_target)

stdv_target = trainset.label:std()
trainset.label:div(stdv_target)
   

-- torch.save('train_normalized.t7', trainset)
--torch.save('test_normalized.t7', testset)


local function inception(input_size, config)
   local concat = nn.Concat(2)
   if config[1][1] ~= 0 then
      local conv1 = nn.Sequential()
      conv1:add(nn.SpatialConvolution(input_size, config[1][1],1,1,1,1)):add(nn.ReLU(true))
      concat:add(conv1)
   end

   local conv3 = nn.Sequential()
   conv3:add(nn.SpatialConvolution(  input_size, config[2][1],1,1,1,1)):add(nn.ReLU(true))
   conv3:add(nn.SpatialConvolution(config[2][1], config[2][2],3,3,1,1,1,1)):add(nn.ReLU(true))
   concat:add(conv3)

   local conv3xx = nn.Sequential()
   conv3xx:add(nn.SpatialConvolution(  input_size, config[3][1],1,1,1,1)):add(nn.ReLU(true))
   conv3xx:add(nn.SpatialConvolution(config[3][1], config[3][2],3,3,1,1,1,1)):add(nn.ReLU(true))
   conv3xx:add(nn.SpatialConvolution(config[3][2], config[3][2],3,3,1,1,1,1)):add(nn.ReLU(true))
   concat:add(conv3xx)

   local pool = nn.Sequential()
   pool:add(nn.SpatialZeroPadding(1,1,1,1)) -- remove after getting nn R2 into fbcode
   if config[4][1] == 'max' then
      pool:add(nn.SpatialMaxPooling(3,3,1,1):ceil())
   elseif config[4][1] == 'avg' then
      pool:add(nn.SpatialMaxPooling(3,3,1,1):ceil())
   else
      error('Unknown pooling')
   end
   if config[4][2] ~= 0 then
      pool:add(nn.SpatialConvolution(input_size, config[4][2],1,1,1,1)):add(nn.ReLU(true))
   end
   concat:add(pool)

   return concat
end



model = nn.Sequential()
model:add(nn.SpatialConvolution(3,64,7,7,2,2,3,3)):add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
--model:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
--model:add(nn.ReLU)
--model:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
model:add(nn.SpatialConvolution(64, 64, 1, 1)):add(nn.ReLU(true))
model:add(nn.SpatialConvolution(64,192,3,3,1,1,1,1)):add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())

 -- 3 input image channels, 6 output channels, 5x5 convolution kernel
--model:add(nn.Dropout(0.25))
--model:add(nn.SpatialConvolution(12, 22, 3, 3))
--model:add(nn.ReLU)
--model:add(nn.SpatialMaxPooling(2,2,2,2))

model:add(inception( 192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}})) -- 3(a)
model:add(inception( 256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}})) -- 3(b)
model:add(inception( 320, {{  0},{128,160},{ 64, 96},{'max',  0}})) -- 3(c)

model:add(nn.SpatialConvolution(576,576,2,2,2,2))


model:add(inception( 576, {{224},{ 64, 96},{ 96,128},{'avg',128}})) -- 4(a)
model:add(inception( 576, {{192},{ 96,128},{ 96,128},{'avg',128}})) -- 4(b)
model:add(inception( 576, {{160},{128,160},{128,160},{'avg', 96}})) -- 4(c)
model:add(inception( 576, {{ 96},{128,192},{160,192},{'avg', 96}})) -- 4(d)

   local main_branch = nn.Sequential()
   main_branch:add(inception( 576, {{  0},{128,192},{192,256},{'max',  0}})) -- 4(e)
   main_branch:add(nn.SpatialConvolution(1024,1024,2,2,2,2))
   main_branch:add(inception(1024, {{352},{192,320},{160,224},{'avg',128}})) -- 5(a)
   main_branch:add(inception(1024, {{352},{192,320},{192,224},{'max',128}})) -- 5(b)
   main_branch:add(nn.SpatialAveragePooling(7,7,1,1))
   main_branch:add(nn.View(1024):setNumInputDims(3))
   main_branch:add(nn.Linear(728,1))

   -- add auxillary classifier here (thanks to Christian Szegedy for the details)
   local aux_classifier = nn.Sequential()
   aux_classifier:add(nn.SpatialMaxPooling(5,5,3,3):ceil())
   aux_classifier:add(nn.SpatialConvolution(576,128,1,1,1,1))
   aux_classifier:add(nn.View(128*4*4):setNumInputDims(3))
   aux_classifier:add(nn.Linear(128*4*4,768))
   aux_classifier:add(nn.ReLU())
   aux_classifier:add(nn.Linear(768,1))

   local splitter = nn.Concat(2)
   splitter:add(main_branch):add(aux_classifier)
   local realmodel = nn.Sequential():add(model):add(splitter)


  local features
  features = nn.Concat(2)


   local fb1 = nn.Sequential() -- branch 1
   fb1:add(nn.SpatialConvolutionMM(3,48,11,11,4,4,2,2))       -- 224 -> 55
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   fb1:add(nn.SpatialConvolutionMM(48,128,5,5,1,1,2,2))       --  27 -> 27
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   fb1:add(nn.SpatialConvolutionMM(128,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialConvolutionMM(192,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialConvolutionMM(192,128,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6




   local fb2 = fb1:clone() -- branch 2
   for k,v in ipairs(fb2:findModules('nn.SpatialConvolutionMM')) do
      v:reset() -- reset branch 2's weights
   end

   features:add(fb1)
   features:add(fb2)

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(128*4*2))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(128*4*2, 4096))
   --classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
--   classifier:add(nn.Linear(4096, 4096))
--   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, 1))

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)



--model:add(nn.View(96 * 3 * 3))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
--model:add(nn.Linear(96, 1))

--model:add(nn.View(192 * 3 * 3))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
--model:add(nn.Linear(192 * 3 * 3, 500))             -- fully connected layer (matrix multiplication between input and weights)
--model:add(nn.Dropout(0.5))
--model:add(nn.Linear(500, 84))
--model:add(nn.Dropout(0.25))
--model:add(nn.Linear(84, 1))                   -- 10 is the number of outputs of the network (in this case, 10 digits)



--model = nn.Sequential()
--model:add(nn.SpatialMaxPooling(2,2,2,2)) 
--model:add(nn.View(16*16*3))
--model:add(nn.Tanh())
----net:add(nn.Linear(768, 10))
--model:add(nn.Linear(768, 1))
----net:add(nn.LogSoftMax())
--

model = model--:cl()
criterion = nn.MSECriterion()--:cl()

trainset.data = trainset.data--:cl()
trainset.label = trainset.label--:cl()

--trainer = nn.StochasticGradient(model, criterion)
--trainer.learningRate = 0.01
--trainer.maxIteration = 50
--trainer:train(trainset)


----[[

x, dl_dx = model:getParameters()

data = trainset.data

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights,
-- which, in this example, are all the weights of the linear matrix of
-- our mode, plus one bias.

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1

   if _nidx_ > (#data)[1] then _nidx_ = 1 end

   local sample = trainset[_nidx_]
   local target = sample[2]      -- this funny looking syntax allows
   local inputs = sample[1]    -- slicing of arrays.

   
   --print("Sample", sample)

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample

   --print("Forward", model:forward(inputs))

   --print("target", target)

   local loss_x = criterion:forward(model:forward(inputs), target)

   --print("Loss", loss_x)

   model:backward(inputs, criterion:backward(model.output, target))

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- Given the function above, we can now easily train the model using SGD.
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic 
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely

sgd_params = {
--   learningRate = 1e-3,
learningRate = 5e-3,
   learningRateDecay = 0,
   weightDecay = 0,
   momentum = 0
}

-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-correlation.

-- we cycle 1e4 times over our training data
for i = 1,1e3 do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,(#data)[1] do
--      for i = 1,5 do

      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific
      
      _, fs = optim.sgd(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / (#data)[1]
   print('current loss = ' .. current_loss)

   amount_check = 1000

   err = torch.Tensor(amount_check)

   for i = 1, amount_check do
      pred = model:forward(trainset.data[i])[1] * stdv_target + mean_target

      print("Pred is", pred)

      real = trainset.label[i][1] * stdv_target + mean_target

      print("Real is", real)

      diff = (pred - real)
   
      print("Diff is", diff)
      print("")

      err[i] = torch.abs(diff)

   end

   print("Total error is", err:mean())

end

--]]--
