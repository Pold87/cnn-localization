require 'camera'
require 'image'
require 'torch'
require 'nn'
require 'csvigo'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Deep Drone - Cam Visualization')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-dev', 0, 'Device number of the camera; on a laptop, 0 is usually the in-built cam, 1 the first external one, etc. ')
cmd:option('-batchNorm', true, 'Use batch normalization')
cmd:text()
opt = cmd:parse(arg or {})

-- Settings
dev = opt.dev
width = 320
height = 240
fps = 30

base_dir = "/home/pold87/Documents/Internship/draug/"

src = image.load("../../data/dice.jpg")
img_folder = base_dir .. "genimgs/"
csv_file = csvigo.load(base_dir .. "targets.csv")


use_opencl = false
max_iterations = 50

-- Amount of synthetic views

----------------------------------------------------------------------
-- training/test size

trsize = 100 -- training images

img_width = 224
img_height = 224

x_range = img_width * 2
y_range = img_height * 2
total_range = 350

-- Read CSV and convert to tensor
csv_file = csvigo.load("../../data/targets.csv")
target_x = torch.Tensor(csv_file.x)
target_y = torch.Tensor(csv_file.y)
target_z = torch.Tensor(csv_file.z)


-- Start by predicting the x coordinate
trainset = {
   data = torch.Tensor(trsize, 3, img_width, img_height),
   label = torch.FloatTensor(trsize, 1, total_range),
   size = function() return trsize end
}


-- Load train data (incl. Gaussian normalization)
function load_data(dataset, start_pic_num, pics)

   i_prime = start_pic_num

   for i = 1, pics do

      img = image.load(img_folder .. (i_prime - 1) .. ".png")

      img = image.scale(img, img_width, img_height)
   
      dataset.data[i] = img
      end
       
   end


load_data(trainset, 1, trsize)

function trainset:size() 
    return self.data:size(1) 
end

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

trainset.data = trainset.data:float() -- convert the data from a ByteTensor to a DoubleTensor.

-- Convert all images to YUV

print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1, trainset:size() do
   trainset.data[i] = image.rgb2yuv(trainset.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}

print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i, channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainset.data[{ {},i,{},{} }]:mean()
   std[i] = trainset.data[{ {},i,{},{} }]:std()
   trainset.data[{ {},i,{},{} }]:add(-mean[i])
   trainset.data[{ {},i,{},{} }]:div(std[i])
end



-- Initialize camera
cam = image.Camera{idx=dev,width=width,height=height,fps=fps}  -- create the camera grabber

-- Grab frame
frame = cam:forward()

-- Resize image
frame = image.scale(frame, 224, 224)

-- Display frame
win = image.display{win=win,image=frame}

model = torch.load("results/model.t7")



-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)
   
-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()


function preprocess(img)

   img = image.scale(img, 224, 224)

   img = img:float()

   img = image.rgb2yuv(img)

   -- Normalize test data, using the training means/stds
   for i,channel in ipairs(channels) do
      -- normalize each channel globally:
      img[{i,{},{} }]:add(-mean[i])
      img[{i,{},{} }]:div(std[i])
   end

   -- Normalize all channels locally:
   for c = 1, 3 do
      img[{ {c},{},{} }] = normalization:forward(img[{ {c},{},{} }])
   end

   return img

end

while true do

   frame = cam:forward():float()  -- return the next frame available

   frame = preprocess(frame)

      -- test sample
   if opt.batchNorm then
      local batchData = torch.Tensor(1, 3, img_width, img_height):float()
      batchData[1] = frame:float()

       pred = model:forward(batchData)
   else
       pred = model:forward(frame)
   end

   max, pos = pred:max(1)

   print(pos)

   image.display{win=win, image=frame}  -- display frame


end

cam:stop() -- release the camera
