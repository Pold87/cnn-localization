require 'nn'
require 'clnn'
require 'optim'
require 'ffmpeg'
require 'image'
require 'csvigo'
require 'qtwidget'

src = image.load("../data/dice.jpg")

amount_pics = 300

-- Read CSV and convert to tensor
csv_file = csvigo.load("../draug/targets.csv")
target_x = torch.Tensor(csv_file.x)


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
   data = torch.Tensor(amount_pics, 3, 224, 224),
   label = torch.DoubleTensor(amount_pics, 224),
   size = function() return amount_pics end
}


-- Load train data

for i=1, amount_pics do

   img = image.load("../draug/genimgs/" .. i .. ".png")

   true_x = target_x[i]
   int_true_x = math.min(math.floor(true_x), 224)

   label = torch.Tensor(224)

   print(int_true_x)

   label[int_true_x] = 1

   trainset.data[i] = img
   trainset.label[i] = label
   --trainset.label[i] = int_true_x
    
end



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


mean_target = torch.DoubleTensor(1) -- store the mean, to normalize the test set in the future
stdv_target  = torch.DoubleTensor(1) -- store the standard-deviation for the future


for i=1, 0 do -- over x, y, height, width
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

--print(trainset.label)

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16* 53 * 53))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16* 53 * 53, 16 * 29))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Linear(16 * 29, 224))             -- fully connected layer (matrix multiplication between input and weights)
-- net:add(nn.Linear(120, 2))2
--net:add(nn.Linear(84, amount_pics))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

-- criterion = nn.MSECriterion()
criterion = nn.BCECriterion()
-- criterion = nn.ClassNLLCriterion()

-- Move to OpenCL

net = net:cl()
criterion = criterion:cl()
trainset.data = trainset.data:cl()
trainset.label = trainset.label:cl()
--mean_target = mean_target:cl()
--stdv_target = stdv_target:cl()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 10

--print(trainset.data[5])

trainer:train(trainset)

-- Testing

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
