require 'torch'
require 'nn'
require 'cunn'
require 'image'
require 'math'
require 'csvigo'
require 'distributions'
require 'gnuplot'
require 'dp'


-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Drone Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-dof', 1, 'degrees of freedom; 1: only x coordinates, 2: x, y; etc.')
   cmd:option('-baseDir', '/home/pold/Documents/draug/', 'Base dir for images and targets')
   cmd:option('-regression', true, 'Base directory for images and targets')
   cmd:option('-standardize', false, 'apply Standardize preprocessing')
   cmd:option('-zca', false, 'apply Zero-Component Analysis whitening')
   cmd:option('-scaleImages', false, 'scale input images to 224 x 224')
   cmd:option('-lecunlcn', true, 'apply Yann LeCun Local Contrast Normalization (recommended)')
   cmd:text()
   opt = cmd:parse(arg or {})
end


-- Settings --

img_folder = opt.baseDir .. "genimgs/"
csv_file = csvigo.load(opt.baseDir .. "targets.csv")

img_width = 224
img_height = 224

-- Amount of synthetic views

----------------------------------------------------------------------
-- training/test size

if opt.size == 'full' then
   print '==> using regular, full training data'
   -- 510 worked perfectly
   trsize = 2500 -- training images
   tesize = 300 -- test images
   totalSize = 1000
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   trsize = 40
   tesize = 40
   totalSize = 50
elseif opt.size == 'xsmall' then
   print '==> using reduced training data, for fast experiments'
   trsize = 8
   tesize = 8
   totalSize = 15
end


if opt.regression then 
   total_range = 1
else  
   total_range = 350
end

-- Convert csv columns to tensors
local target_x = torch.Tensor(csv_file.x)
local target_y = torch.Tensor(csv_file.y)
local target_z = torch.Tensor(csv_file.z)


function visualize_data(targets)

   print(targets)
   hist = torch.histc(targets, 10)
   gnuplot.hist(targets, 10, 1, 10)
   print("Histogram", hist)

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


function load_data_dp(dataPath, validRatio)

   local input = torch.CudaTensor(totalSize, 3, img_height, img_width)
   local target = torch.CudaTensor(totalSize)

   for i = 1, totalSize do

      local img = image.load(dataPath .. "genimgs/" .. (i - 1) .. ".png")
      input[i] = image.rgb2yuv(img)
      target[i] = target_x[i]
      collectgarbage()
   end

   local nValid = math.floor(totalSize * validRatio)
   local nTrain = totalSize - nValid

   local trainInput = dp.ImageView('bchw', input:narrow(1, 1, nTrain))
   local trainTarget = dp.DataView('b', target:narrow(1, 1, nTrain))

   local validInput = dp.ImageView('bchw', input:narrow(1, nTrain+1, nValid))
   local validTarget = dp.DataView('b', target:narrow(1, nTrain+1, nValid))

   -- 3. wrap views into datasets

   local train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
   local valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}

   -- 4. wrap datasets into datasource

   local ds = dp.DataSource{train_set=train,valid_set=valid}

   return ds
end

--[[data]]--
ds = load_data_dp(opt.baseDir, 0.2)
st = dp.Standardize()

trainTargets = ds:trainSet():targets()
validTargets = ds:validSet():targets()

trainInputs = ds:trainSet():inputs()
validInputs = ds:validSet():inputs()

st:apply(trainTargets, true)
st:apply(validTargets, false)

--[[preprocessing]]--
local input_preprocess = {}
if opt.standardize then
   table.insert(input_preprocess, dp.Standardize())
   staImages = dp.Standardize()
   staImages:apply(trainInputs, true)
   staImages:apply(validInputs, false)
end
if opt.zca then
   table.insert(input_preprocess, dp.ZCA())

   zca = dp.ZCA()
   zca:apply(trainInputs)
   zca:apply(validInputs)
end
if opt.lecunlcn then
   table.insert(input_preprocess, dp.GCN())
   table.insert(input_preprocess, dp.LeCunLCN{progress=true})

   gcn = dp.GCN()
   lec = dp.LeCunLCN{progress=true}

   gcn:apply(trainInputs)
   gcn:apply(validInputs)
   
end
