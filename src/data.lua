require 'ffmpeg'
require 'image'
require 'csvigo'

-- Read CSV and convert to tensor
csv_file = csvigo.load("data.csv")
gps_x = torch.Tensor(csv_file.gps_x)

dspath = '/home/pold/Documents/torchstuff/video.avi'
source = ffmpeg.Video{path=dspath,
--                      width=32, height=32, 
                      width=284/2, height=160/2, 
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

trainData = {
   data = torch.Tensor(trainImaNumber, ivch, ivhe, ivwi),
   label = torch.Tensor(trainImaNumber, 1),
   size = function() return trainImaNumber end
}

testData = {
   data = torch.Tensor(trainImaNumber, ivch, ivhe, ivwi),
   label = torch.Tensor(trainImaNumber, 1),
   size = function() return trainImaNumber end
}


-- Split the data into train and test images

-- load train data: 2416 images
-- for i = 0, trainImaNumber do
for i = 0, trainImaNumber - 1 do

  -- get next frame:
  img = source:forward()

  i_prime = math.floor(i / 2)

  if i % 2 == 0 then
  -- Decide if it goes to train or test data
     trainData.data[i_prime + 1] = img
     trainData.label[i_prime + 1] = gps_x[i + 1]
     print(trainData.label[i_prime + 1])

  else
     testData.data[i_prime + 1] = img
     testData.label[i_prime + 1] = gps_x[i + 1]
     print(testData.label[i_prime + 1])
  end

end

--image.display{image=trainData.data[{{500,550}}], 
--              legend = 'Train Data'}

-- print(trainData.labels[200])

-- Save data

torch.save('train.t7', trainData)
torch.save('test.t7', testData)
