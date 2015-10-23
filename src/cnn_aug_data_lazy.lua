require 'dp'
require 'torchx' -- for paths.indexdir

function uavestimation(dataPath, validRatio)
   validRatio = validRatio or 0.15

   -- 1. load images into input and target Tensors
   local imgs = paths.indexdir(paths.concat(dataPath, 'genimgs')) -- 1
   local size = imgs:size()
   local shuffle = torch.randperm(size) -- shuffle the data
   local input = torch.FloatTensor(size, 3, 112, 112)
   local target = torch.IntTensor(size):fill(2)

   for i=1, bg:size() do
      local img = image.load(imgs:filename(i))
      local idx = shuffle[i]
      input[idx]:copy(img)
      target[idx] = 1
      collectgarbage()
   end

   -- 2. divide into train and valid set and wrap into dp.Views

   local nValid = math.floor(size * validRatio)
   local nTrain = size - nValid

   local trainInput = dp.ImageView('bchw', input:narrow(1, 1, nTrain))
   local trainTarget = dp.ClassView('b', target:narrow(1, 1, nTrain))
   local validInput = dp.ImageView('bchw', input:narrow(1, nTrain + 1, nValid))
   local validTarget = dp.ClassView('b', target:narrow(1, nTrain + 1, nValid))

   trainTarget:setClasses({'bg', 'face'})
   validTarget:setClasses({'bg', 'face'})

   -- 3. wrap views into datasets

   local train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
   local valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}

   -- 4. wrap datasets into datasource

   local ds = dp.DataSource{train_set=train,valid_set=valid}
   ds:classes{'bg', 'face'}
   return ds
end
