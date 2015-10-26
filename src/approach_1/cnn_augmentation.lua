require 'nn'
require 'nnx'
require 'math'
require 'clnn'
require 'optim'
require 'ffmpeg'
require 'image'
require 'csvigo'
require 'qtwidget'
require 'dp'

-- Settings
src = image.load("../data/dice.jpg")
img_folder = "../../draug/genimgs/"
use_opencl = false
max_iterations = 50

-- Amount of synthetic views
amount_pics = 300
amount_test_pics = 70
img_width = 224 / 2
img_height = 224 / 2

-- Read CSV and convert to tensor
csv_file = csvigo.load("../../draug/targets.csv")
target_x = torch.Tensor(csv_file.x)
target_y = torch.Tensor(csv_file.y)
target_z = torch.Tensor(csv_file.z)


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


function makeTargets(y, stdv)
   -- y : (batch_size, num_keypoints*2)
   -- Y : (batch_size, num_keypoints*2, 98)
   Y = torch.FloatTensor(y:size(1), y:size(2), total_range):zero()
   pixels = torch.range(1,total_range):float()
   local k = 0
   for i=1,y:size(1) do
      local keypoints = y[i]
      local new_keypoints = Y[i]
      for j=1,y:size(2) do
         local kp = keypoints[j]
         if kp ~= -1 then
            local new_kp = new_keypoints[j]
            new_kp:add(pixels, -kp)
            new_kp:cmul(new_kp)
            new_kp:div(2 * stdv * stdv)
            new_kp:mul(-1)
            new_kp:exp(new_kp)
            new_kp:div(math.sqrt(2 * math.pi) * stdv)
         else
            k = k + 1
         end
      end
   end
   return Y
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
y_range = img_height * 2
total_range = 300

-- Start by predicting the x coordinate
trainset = {
   data = torch.Tensor(amount_pics, 3, img_width, img_height),
   label = torch.DoubleTensor(amount_pics, 2, total_range),
   size = function() return amount_pics end
}

-- Start by predicting the x coordinate
testset = {
   data = torch.Tensor(amount_test_pics, 3, img_width, img_height),
   label = torch.DoubleTensor(amount_test_pics, 2, total_range),
   size = function() return amount_test_pics end
}


-- Load train data (incl. Gaussian normalization)
function load_data(dataset, start_pic_num, pics)

   label = torch.Tensor(pics, 2)

   i_prime = start_pic_num

   for i = 1, pics do

      img = image.load(img_folder .. (i_prime - 1) .. ".png")

      img = image.scale(img, img_width, img_height)
   
      true_x = target_x[i_prime]
      true_x = true_x + 112
      int_true_x = math.min(math.floor(true_x),  total_range)
      

      true_y = target_y[i_prime]
      true_y = true_y + 112
      int_true_y = math.min(math.floor(true_y),  total_range)
      
      print("i_prime is",i_prime, i)
      dataset.data[i] = img
      

      label[i][1] = true_x
      label[i][2] = true_y

      i_prime = i_prime + 1
      
   end

   dataset.label = makeTargets(label, 1)


end

load_data(trainset, 1, amount_pics)
load_data(testset, 201, amount_test_pics)


-- Load test data with using Gaussian normalization
function load_data_raw(start_pic_num, end_pic_num)

   for i = 1, amount_pics do

      img = image.load(img_folder .. (i - 1) .. ".png")
      
      img = image.scale(img, img_width, img_height)
      
      true_x = target_x[i]
      true_x = true_x + 112
      int_true_x = math.min(math.floor(true_x),  total_range)
      
      
      true_y = target_y[i]
      true_y = true_y + 112
      int_true_y = math.min(math.floor(true_y),  total_range)
      
      
      --true_z = target_z[i]
      --int_true_z = math.min(math.floor(true_z),  total_range)
      
      
      label = torch.Tensor(2, total_range):fill(0)
      
      label[1][int_true_x] = 1
      label[2][int_true_y] = 1
      --label[3][int_true_z] = 1
      
      
      trainset.data[i] = img
      trainset.label[i] = label
      --trainset.label[i] = int_true_x
      
   end

end
   



-- for i = 1, amount_test_pics do
-- 
--    i_prime = amount_pics + i
-- 
--    img = image.load(img_folder .. (i_prime - 1) .. ".png")
-- 
--    img = image.scale(img, img_width, img_height)
-- 
--    true_x = target_x[i_prime]
--    true_x = true_x + 112
--    int_true_x = math.min(math.floor(true_x),  x_range)
-- 
-- 
--    true_y = target_y[i_prime]
--    true_y = true_y + 112
--    int_true_y = math.min(math.floor(true_y),  total_range)
-- 
-- 
--    true_z = target_z[i_prime]
--    int_true_z = math.min(math.floor(true_z),  total_range)
-- 
-- 
--    label = torch.Tensor(2, total_range):fill(0)
-- 
--    label[1][int_true_x] = 1
--    label[2][int_true_y] = 1
--    --label[3][int_true_z] = 1
-- 
-- 
--    --print("True x is " .. true_x .. "... and rounded " .. int_true_x)
--    
--    testset.data[i] = img
--    testset.label[i] = label
--     
-- end


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


-- TESTSET

mean_test = {} -- store the mean, to normalize the test set in the future
stdv_test  = {} -- store the standard-deviation for the future
for i=1, 3 do -- over each image channel
    mean[i] = testset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = testset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
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


function testset:size() 
    return self.data:size(1) 
end

setmetatable(testset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

testset.data = testset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.


print(testset.label:size())
print(trainset.label:size())

--print(trainset.label)

input_size = 3 * img_width * img_height

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5))
net:add(nn.SpatialMaxPooling(2, 2, 2, 2)) 
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16 * 25 * 25))
net:add(nn.Linear(16 * 25 * 25, 2 * total_range))
net:add(nn.Reshape(2,total_range))
--net:add(nn.LogSoftMax())
net:add(nn.MultiSoftMax())


logModule = nn.Sequential()
logModule:add(nn.AddConstant(0.00000001)) -- fixes log(0)=NaN errors
logModule:add(nn.Log())

criterion = nn.ModuleCriterion(nn.DistKLDivCriterion(), logModule, nn.Convert())

-- Move to OpenCL
if use_opencl then
   net = net:cl()
   criterion = criterion:cl()
   trainset.data = trainset.data:cl()
   trainset.label = trainset.label:cl()
   mean_target = mean_target:cl()
   stdv_target = stdv_target:cl()
end

--trainer = nn.StochasticGradient(net, criterion)
--trainer.learningRate = 0.001
--trainer.maxIteration = max_iterations

--trainer:train(trainset)

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


-- Save the trained model
-- torch.save("trained_model.t7", net)

function visual_comparison(ground_truth, preds)

   
   local ground_truth_normalized = ground_truth * 255
   local preds_normalized = preds * 255

   local height = 100
   local width_per_cell = 5

   -- local 2d_truth = torch.Tensor(height, preds:size() * width_per_cell):fill(0)
   local sep_2d = torch.Tensor(10, preds:size(1)):fill(130)
   -- local 2d_preds = torch.Tensor(height, preds:size() * width_per_cell):fill(0)

   truth_2d = torch.repeatTensor(ground_truth_normalized, height, 1)
   preds_2d = torch.repeatTensor(preds_normalized, height, 1)

   pic = torch.cat({truth_2d, sep_2d, preds_2d}, 1)

   image.display(pic)

end


--[[for i = 1, 20 do

   estimations = net:forward(trainset.data[i])

   print("Ground truth", trainset.label[i])
   print("Estimations", estimations:exp())

   visual_comparison(trainset.label[i], estimations:exp())
   sleep(1.5)
   
end
--]]

--[[for i = 1, 20 do

   estimations = net:forward(testset.data[i])

   print("Ground truth", testset.label[i])
   print("Estimations", estimations[1]:exp())

   visual_comparison(testset.label[i][1], estimations[1]:exp())
   -- sleep(1.5)
   
end
--]]
