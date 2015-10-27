require 'camera'
require 'image'
require 'torch'
require 'nn'
require 'csvigo'

-- Settings
dev = 1
width = 320
height = 240
fps = 30


src = image.load("../../data/dice.jpg")
img_folder = "../../data/genimgs/"
use_opencl = false
max_iterations = 50

-- Amount of synthetic views

----------------------------------------------------------------------
-- training/test size

trsize = 100 -- training images
tesize = 100 -- test images

img_width = 224 / 2
img_height = 224 / 2

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

-- Start by predicting the x coordinate
testset = {
   data = torch.Tensor(tesize, 3, img_width, img_height),
   label = torch.FloatTensor(tesize, 1, total_range),
   size = function() return tesize end
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
frame = image.scale(frame, 112, 112)

-- Display frame
win = image.display{win=win,image=frame}

model = torch.load("results/model.t7")

while true do

   frame = cam:forward():float()  -- return the next frame available

   frame = image.scale(frame, 112, 112)

   frame = frame:float()

   frame = image.rgb2yuv(frame)

   -- Normalize test data, using the training means/stds
   for i,channel in ipairs(channels) do
      -- normalize each channel globally:
      frame[{i,{},{} }]:add(-mean[i])
      frame[{i,{},{} }]:div(std[i])
   end

   -- Define the normalization neighborhood:
   neighborhood = image.gaussian1D(13)
   
   -- Define our local normalization operator (It is an actual nn module, 
   -- which could be inserted into a trainable model):
   normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
   
   -- Normalize all channels locally:
   for c = 1, 3 do
	 frame[{ {c},{},{} }] = normalization:forward(frame[{ {c},{},{} }])
      end

   
   pred = model:forward(frame)
   max, pos = pred:max(1)

   image.display{win=win, image=frame}  -- display frame


   print(pos[1])

end

cam:stop() -- release the camera
