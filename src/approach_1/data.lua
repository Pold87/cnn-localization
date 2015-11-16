require 'torch'
require 'nn'
require 'image'
require 'math'
require 'csvigo'
require 'distributions'
require 'gnuplot'


-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Drone Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-dof', 1, 'degrees of freedom; 1: only x coordinates, 2: x, y; etc.')
   cmd:option('-baseDir', '/home/pold/Documents/draug/', 'Base dir for images and targets')
   cmd:option('-regression', true, 'Base directory for images and targets')
   cmd:text()
   opt = cmd:parse(arg or {})
end


-- Settings

base_dir = opt.baseDir

-- Folder of draug images
img_folder = base_dir .. "genimgs/"
csv_file = csvigo.load(base_dir .. "targets.csv")

----------------------------------------------------------------------
-- training/test size (Amount of synthetic views)

if opt.size == 'full' then
   print '==> using regular, full training data'
   trsize = 2500 -- training images
   tesize = 300 -- test images
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   trsize = 40
   tesize = 40
elseif opt.size == 'xsmall' then
   print '==> using reduced training data, for extremely fast experiments'
   trsize = 8
   tesize = 8
end

img_width = 224
img_height = 224

if opt.regression then 
   total_range = 1
else  
   total_range = 350
end

-- Convert csv columns to tensors
target_x = torch.Tensor(csv_file.x)
target_y = torch.Tensor(csv_file.y)
target_z = torch.Tensor(csv_file.z)


trainset = {
   data = torch.Tensor(trsize, 3, img_width, img_height),
   label = torch.FloatTensor(trsize, opt.dof, total_range),
   size = function() return trsize end
}

testset = {
   data = torch.Tensor(tesize, 3, img_width, img_height),
   label = torch.FloatTensor(tesize, opt.dof, total_range),
   size = function() return tesize end
}


-- These are some helper functions that are able to convert
-- standardized targets to raw targets They are needed because the MSE
-- of the normal targets gets otherwise too big and the network does
-- not train.
function normalized_to_raw(pred, mean_target, stdv_target)
    
    val = pred:clone()
    val = val:cmul(stdv_target)
    val = val:add(mean_target)
    
    return val 
end


function normalized_to_raw_num(pred, mean_target, stdv_target)
    
    val = pred
    val = val * stdv_target
    val = val + mean_target
    
    return val 
end


function raw_to_normalized(pred, mean_target, stdv_target)
    
    val = pred:clone()
    val = val:add(- mean_target)
    val = val:cdiv(stdv_target)
    
    return val 
end


function visualize_data(targets)

   print(targets)
   hist = torch.histc(targets, 10)
   gnuplot.hist(targets, 10, 1, 10)
   print("Histogram", hist)

end



function sleep(n)
  os.execute("sleep " .. tonumber(n))
end



function set_small_nums_to_zero(x)
   val = 0
   if x > 0.01 then
      val = x
   end   
   return val
end      

function makeTargets(y, stdv)

   mean_pos = y / total_range
   
   Y = image.gaussian1D({size=total_range,
			 mean=mean_pos,
			 sigma=.0035,
			 normalize=true})
   Y:apply(set_small_nums_to_zero)

   return Y

end

-- Load train data (incl. Gaussian normalization for classification)
function load_data(dataset, start_pic_num, pics)

   -- Specify degrees of freedom
   if opt.dof == 1 then
      label = torch.Tensor(pics)
   else
      label = torch.Tensor(pics, opt.dof)
   end

   -- TODO: think about if a while loop is better here
   i_prime = start_pic_num

   for i = 1, pics do

      -- Load image from folder
      img = image.load(img_folder .. (i_prime - 1) .. ".png")

      -- Scale to desired size
      img = image.scale(img, img_width, img_height)
   
      -- Get coordinates from csv file
      true_x = target_x[i_prime]
      true_y = target_y[i_prime]

      -- Make sure that all true values are between 0 and total_range
      if not opt.regression then
	 true_x = math.min(math.floor(true_x),  total_range)
	 true_y = math.min(math.floor(true_y),  total_range)
      end
     
      -- Set dataset
      dataset.data[i] = img

      -- Degrees of freedom
      if opt.dof == 1 then
         label[i] = true_x
      else
         label[i][1] = true_x
      end
      if opt.dof >= 2 then
         label[i][2] = true_y
      end

      i_prime = i_prime + 1

      
      if opt.regression then
	 dataset.label[i] = true_x
      else
	 -- Add Gaussian noise to classification
	 dataset.label[i] = makeTargets(true_x, .15)

      end
       
   end

end

-- Actually load data
load_data(trainset, 1, trsize)
load_data(testset, trsize + 1, tesize)


function trainset:size() 
    return self.data:size(1) 
end

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);


function testset:size() 
    return self.data:size(1) 
end

setmetatable(testset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);


----------------------------------------------------------------------
print '==> preprocessing data'

trainset.data = trainset.data:float() -- convert the data from a ByteTensor to a DoubleTensor.
testset.data = testset.data:float() -- convert the data from a ByteTensor to a DoubleTensor.
trainset.label = trainset.label:float() -- convert the data from a ByteTensor to a DoubleTensor.
testset.label = testset.label:float() -- convert the data from a ByteTensor to a DoubleTensor.


-- Convert all images to YUV

print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1, trainset:size() do
   trainset.data[i] = image.rgb2yuv(trainset.data[i])
end

for i = 1, testset:size() do
   testset.data[i] = image.rgb2yuv(testset.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.

print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}

mean_target = trainset.label:mean()
std_target = trainset.label:std()


for i, channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainset.data[{ {},i,{},{} }]:mean()
   std[i] = trainset.data[{ {},i,{},{} }]:std()

   trainset.data[{ {},i,{},{} }]:add(-mean[i])
   trainset.data[{ {},i,{},{} }]:div(std[i])

end

trainset.label:add(-mean_target)
trainset.label:div(std_target)


-- Normalize test data, using the training means/stds
for i, channel in ipairs(channels) do
   -- normalize each channel globally:
   testset.data[{ {},i,{},{} }]:add(-mean[i])
   testset.data[{ {},i,{},{} }]:div(std[i])

end

testset.label:add(-mean_target)
testset.label:div(std_target)

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1, trainset:size() do
      trainset.data[{ i,{c},{},{} }] = normalization:forward(trainset.data[{ i,{c},{},{} }])
   end
   for i = 1, testset:size() do
      testset.data[{ i,{c},{},{} }] = normalization:forward(testset.data[{ i,{c},{},{} }])
   end
end


-- Take a 1D-tensor (e.g. with size 300), and split it into classes
-- For example, 1-30: class 1; 31 - 60: class 2; etc.
function to_classes(predictions, classes) 

   if opt.regression then

      width = 35
      pos = predictions[1]
      pos = normalized_to_raw_num(pos, mean_target, std_target)

   else
      len = predictions:size()
      max, pos = predictions:max(1)
      pos = pos[1]
      width = len[1] / classes -- width of the bins
   end

   class = (math.floor((pos - 1) / width)) + 1

   return math.min(math.max(class, 1), classes)
   

end

function all_classes(labels, num_classes)
  s = labels:size(1)
  tmp_classes = torch.Tensor(s):fill(0)

  for i=1, labels:size(1) do
     if opt.regression then
	class = to_classes(labels[i][1], 10)  
     else
	class = to_classes(labels[i][1], 10)  
     end
    tmp_classes[i] = class
  end

  return tmp_classes
  
end

function all_classes_2d(labels, num_classes)
  s = labels:size(1)
  tmp_classes = torch.Tensor(s):fill(0)

  for i=1, labels:size(1) do
    class = to_classes(labels[i], 10)  
    tmp_classes[i] = class
  end

  return tmp_classes
  
end


