require 'camera'
require 'image'
require 'torch'
require 'nn'
require 'csvigo'
require 'math'



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

trsize = 100 -- training images

img_width = 224
img_height = 224

x_range = img_width * 2
y_range = img_height * 2
total_range = 350


base_dir = "/home/pold/Documents/draug/"

-- Image folder
img_folder = base_dir .. "genimgs/"


-- Read CSV and convert to tensor
csv_file = csvigo.load(base_dir .. "targets.csv")
target_x = torch.Tensor(csv_file.x)

model = torch.load("results/model.t7")


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




function sleep(n)
  os.execute("sleep " .. tonumber(n))
end


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



-- Display frame
win = image.display{win=win,image=image.load(img_folder .. 0 .. ".png")}

-- Compare images

for i = 1, 1000 do

   img = image.load(img_folder .. (i - 1) .. ".png")

   image.display{win=win, image=img}  -- display frame

   img = preprocess(img)


   print("Actual value is", target_x[i])

   -- test sample
   if opt.batchNorm then
      local batchData = torch.Tensor(1, 3, img_width, img_height):float()
      batchData[1] = img:float()

       pred = model:forward(batchData)
   else
       pred = model:forward(img)
   end


   max, pos = pred:max(2)

   print("Iteration", i)
   print("Predicted value is", pos[1][1])

   print("Diff is", math.abs(pos[1][1] - target_x[i]))
   print("")

end
