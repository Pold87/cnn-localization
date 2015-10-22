require 'nn'
require 'nnx'
require 'clnn'
require 'optim'
require 'ffmpeg'
require 'image'
require 'csvigo'
require 'qtwidget'

-- Settings
src = image.load("../data/dice.jpg")
img_folder = "../../draug/genimgs/"
use_opencl = false
max_iterations = 1

-- Amount of synthetic views
amount_pics = 50
img_width = 224 / 2
img_height = 224 / 2

-- Read CSV and convert to tensor
csv_file = csvigo.load("../../draug/targets.csv")
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

x_range = img_width * 2

-- Start by predicting the x coordinate
trainset = {
   data = torch.Tensor(amount_pics, 3, img_width, img_height),
   label = torch.DoubleTensor(amount_pics, x_range),
   size = function() return amount_pics end
}

-- Load train data

for i=1, amount_pics do

   img = image.load(img_folder .. (i - 1) .. ".png")

   img = image.scale(img, img_width, img_height)

   true_x = target_x[i]
   int_true_x = math.min(math.floor(true_x),  x_range)

   label = torch.Tensor(x_range):fill(0)

   label[int_true_x] = 1

   --print("True x is " .. true_x .. "... and rounded " .. int_true_x)
   
   trainset.data[i] = img
   trainset.label[i] = label
   --trainset.label[i] = int_true_x
    
end


mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1, 3 do -- over each image channel
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

input_size = 3 * img_width * img_height

net = nn.Sequential()
net:add(nn.View(input_size))
net:add(nn.Linear(input_size, x_range))
net:add(nn.LogSoftMax())
criterion = nn.DistKLDivCriterion()

-- Move to OpenCL
if use_opencl then
   net = net:cl()
   criterion = criterion:cl()
   trainset.data = trainset.data:cl()
   trainset.label = trainset.label:cl()
   mean_target = mean_target:cl()
   stdv_target = stdv_target:cl()
end

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = max_iterations

trainer:train(trainset)

-- Testing

--[[
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
--]]



for i = 1, 10 do

   estimations = net:forward(trainset.data[i])

   print("Ground truth", trainset.label[i])
   print("Estimations", estimations:exp())
   
end
