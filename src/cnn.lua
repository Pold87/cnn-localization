require 'nn'
require 'clnn'
require 'optim'
require 'ffmpeg'
require 'image'
require 'csvigo'
require 'qtwidget'

src = image.load("wallpaper.jpg")
amount_pics = 300

function normalized_to_raw(pred, mean_target, stdv_target)
    
    val = pred:clone()
    val = val:cmul(stdv_target)
    val = val:add(mean_target)
    
    return val 
end

function raw_to_normalized(pred, mean_target, stdv_target)
    
    val = pred:clone()
    val = val:add(- mean_target)
    val = val:cdiv(stdv_target)
    
    return val 
end

function extract_one(i)
    
    width = 128
    height = 128
    --print(src:size())
    x1 = torch.random(src:size()[3] - width)
    --print(x1)
    x2 = x1 + width
    --print(x2)
    y1 = torch.random(src:size()[2] - height)
    --print(y1)

    y2 = y1 + height
    --print(y2)
    i_c = image.crop(src, x1, y1, x2, y2)
    
    top_left = torch.Tensor({x1, y1})
    bottom_right = torch.Tensor({x1 + width, y1 + height})
    
    return i_c, x1, y1, width, height
end


function sleep(n)
  os.execute("sleep " .. tonumber(n))
end


function display_box(img, x1, y1, x2, y2, width, height)

    -- Display image and get handle
    win = qtwidget.newwindow(img:size(3), img:size(2))
    image.display{image = img, win = win}

    -- ground truth
    win:setcolor(1,0,0)
    win:rectangle(x1, y1, width, height)
    win:stroke()
    
    -- estimation
    win:setcolor(0,1,0)
    win:rectangle(x2, y2, width, height)
    win:stroke()
    
    return win
end

trainset = {
   data = torch.Tensor(amount_pics, 3, 128, 128),
   label = torch.DoubleTensor(amount_pics, 2),
   size = function() return amount_pics end
}


for i=1, amount_pics do
    i_c, x1, y1, width, height = extract_one(src)
    
    trainset.data[i] = i_c
    
    print(x1,y1, width,height)
    
    -- Remove width and height estimations for now
    --label = torch.Tensor({x1, y1, width, height})
    label = torch.Tensor({x1, y1})
    
    trainset.label[i] = label
    
end



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


-- model:add(inception( 192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}})) -- 3(a)
-- model:add(inception( 256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}})) -- 3(b)
-- model:add(inception( 320, {{  0},{128,160},{ 64, 96},{'max',  0}})) -- 3(c)
-- 
-- model:add(nn.SpatialConvolution(576,576,2,2,2,2))
-- 
-- 
-- model:add(inception( 576, {{224},{ 64, 96},{ 96,128},{'avg',128}})) -- 4(a)
-- model:add(inception( 576, {{192},{ 96,128},{ 96,128},{'avg',128}})) -- 4(b)
-- model:add(inception( 576, {{160},{128,160},{128,160},{'avg', 96}})) -- 4(c)
-- model:add(inception( 576, {{ 96},{128,192},{160,192},{'avg', 96}})) -- 4(d)
-- 
-- local main_branch = nn.Sequential()
-- main_branch:add(inception( 576, {{  0},{128,192},{192,256},{'max',  0}})) -- 4(e)
-- main_branch:add(nn.SpatialConvolution(1024,1024,2,2,2,2))
-- main_branch:add(inception(1024, {{352},{192,320},{160,224},{'avg',128}})) -- 5(a)
-- main_branch:add(inception(1024, {{352},{192,320},{192,224},{'max',128}})) -- 5(b)
-- main_branch:add(nn.SpatialAveragePooling(7,7,1,1))
-- main_branch:add(nn.View(1024):setNumInputDims(3))
-- main_branch:add(nn.Linear(728,1))
-- 

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


mean_target = torch.DoubleTensor(2) -- store the mean, to normalize the test set in the future
stdv_target  = torch.DoubleTensor(2) -- store the standard-deviation for the future


for i=1, 2 do -- over x, y, height, width
    mean_target[i] = trainset.label[{{}, {i}}]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean_target[i])
    trainset.label[{{}, {i}}]:add(- mean_target[i]) -- mean subtraction
    
    stdv_target[i] = trainset.label[{{}, {i}}]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv_target[i])
    trainset.label[{{}, {i}}]:div(stdv_target[i]) -- std scaling
end

function trainset:size() 
    return self.data:size(1) 
end

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end


net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*29*29))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*29*29, 16 * 29))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Linear(16 * 29, 2))             -- fully connected layer (matrix multiplication between input and weights)
-- net:add(nn.Linear(120, 2))2
--net:add(nn.Linear(84, amount_pics))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
--net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems
-- net:add(nn.Linear(84, 2))                   -- 10 is the number of outputs of the network (in this case, 10 digits)

criterion = nn.MSECriterion()

-- Move to OpenCL

net = net:cl()
criterion = criterion:cl()
trainset.data = trainset.data:cl()
trainset.label = trainset.label:cl()
mean_target = mean_target:cl()
stdv_target = stdv_target:cl()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 10

trainer:train(trainset)

-- Testin

for i = 1, 10 do
    
    -- Ground truth
    i_c, x1, y1, width, height = extract_one(src)
    
    -- Estimation
    pred = net:forward(i_c:cl())
    pred = normalized_to_raw(pred, mean_target, stdv_target)
    
    win = display_box(src, x1, x2, pred[1], pred[2], 128, 128)
    sleep(1.5)
    win:close()
end
