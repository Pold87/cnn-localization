require 'dp'

local UAVCoordinates, DataSource = torch.class("dp.UAVCoordinates", "dp.DataSource")
UAVCoordinates.isUAVCoordinates = true

UAVCoordinates._name = 'UAVCoordinates'
UAVCoordinates._image_size = {1, 112, 112}
UAVCoordinates._feature_size = 3 * 112 * 1112
UAVCoordinates._image_axes = 'bchw'
UAVCoordinates._target_axes = 'bwc'


-- TODO: See if I can omit this and use the default DataSource initalizer
function UAVCoordinates:__init(config) 
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, load_all
   args, self._valid_ratio, self._train_file, self._test_file, 
      self._data_path, self._stdv, self._scale, 
      self._shuffle, load_all = xlua.unpack(
      {config},
      'UAVCoordinates', 
      'https://www.kaggle.com/c/facial-keypoints-detection/data',
      {arg='valid_ratio', type='number', default=1/6,
       help='proportion of training set to use for cross-validation.'},
      {arg='train_file', type='string', default='train.th7',
       help='name of training file'},
      {arg='test_file', type='string', default='test.th7',
       help='name of test file'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
       help='path to data repository'},
      {arg='stdv', type='number', default=0.8, 
       help='standard deviation of the gaussian blur used for targets'},
      {arg='scale', type='table', 
       help='bounds to scale the values between. [Default={0,1}]'},
      {arg='shuffle', type='boolean', 
       help='shuffle train set', default=true},
      {arg='load_all', type='boolean', 
       help='Load all datasets : train, valid, test.', default=true},
      {arg='input_preprocess', type='table | dp.Preprocess',
       help='to be performed on set inputs, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'},
      {arg='target_preprocess', type='table | dp.Preprocess',
       help='to be performed on set targets, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'}  
   )
   self._scale = self._scale or {0,1}
   self._pixels = torch.range(0,111):float()
   if load_all then
      self:loadTrain()
      self:loadValid()
      self:loadTest()
   end
   DataSource.__init(self, {
      train_set=self:trainSet(), 
      valid_set=self:validSet(),
      test_set=self:testSet()
   })
end


function UAVCoordinates:loadTrain()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   local data = self:loadData(self._train_file)
   local start = 1
   local size = math.floor(data:size(1) * (1 - self._valid_ratio))
   local train_data = data:narrow(1, start, size)
   self:trainSet(self:createTrainSet(train_data, 'train'))
   return self:trainSet()
end


function UAVCoordinates:loadValid()
   local data = self:loadData(self._train_file)
   if self._valid_ratio == 0 then
      print"Warning : No Valid Set due to valid_ratio == 0"
      return
   end
   local start = math.ceil(data:size(1)*(1-self._valid_ratio))
   local size = data:size(1) - start
   local valid_data = data:narrow(1, start, size)
   self:validSet(self:createTrainSet(valid_data, 'valid'))

   return self:validSet()
end

function UAVCoordinates:loadData(file_name)
   return torch.load(file_name)
end


function UAVCoordinates:createTrainSet(data, which_set)

   if self._shuffle then
      data = data:index(1, torch.randperm(data:size(1)):long())
   end
   local inputs = data:narrow(2, 7, 112 * 112):clone():view(data:size(1), 1, 112, 112)
   local targets = self:makeTargets(data:narrow(2, 1, 6))

   if self._scale then
      DataSource.rescale(inputs, self._scale[1], self._scale[2])
   end

   -- construct inputs and targets dp.Views 
   local input_v, target_v = dp.ImageView(), dp.SequenceView()
   input_v:forward(self._image_axes, inputs)
   target_v:forward(self._target_axes, targets)

   -- construct dataset
   return dp.DataSet{inputs = input_v,
                     targets = target_v,
                     which_set = which_set}
end


function UAVCoordinates:makeTargets(y)
   -- y : (batch_size, num_keypoints*2)
   -- Y : (batch_size, num_keypoints*2, 98)
   Y = torch.FloatTensor(y:size(1), y:size(2), 112):zero()
   local pixels = self._pixels
   local stdv = self._stdv
   local k = 0
   for i=1, y:size(1) do
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


local FeedbackUAV, parent = torch.class("dp.FeedbackUAV", "dp.Feedback")
FeedbackUAV.isFeedbackUAV = true

function FeedbackUAV:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, precision, name = xlua.unpack(
      {config},
      'FeedbackUAV', 
      'Uses mean square error to measure error w.r.t targets.'..
      'Optionaly compares this to constant (mean) value baseline',
      {arg='precision', type='number', req=true,
       help='precision (an integer) of the keypoint coordinates'},
      {arg='name', type='string', default='uavcoordinate',
       help='name identifying Feedback in reports'}
   )
   config.name = name

   self._precision = precision
   parent.__init(self, config)
   self._pixels = torch.range(0,precision-1):float():view(1,1,precision)
   self._output = torch.FloatTensor()
   self._keypoints = torch.FloatTensor()
   self._targets = torch.FloatTensor()
   self._sum = torch.Tensor():zero()
   self._count = torch.Tensor():zero()
   self._mse = torch.Tensor()
end

function FeedbackUAV:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function FeedbackUAV:doneEpoch(report)
   if self._n_sample > 0 then
      local msg = self._id:toString().." MSE = "..self:meanSquareError()
      print(msg)
   end
end

function FeedbackUAV:meanSquareError()
   return self._mse:cdiv(self._sum, self._count):mean()
end

function FeedbackUAV:_reset()
   self._sum:zero()
   self._count:zero()
end

function FeedbackUAV:report()
   return { 
      [self:name()] = {
         mse = self._n_sample > 0 and self:meanSquareError() or 0
      },
      n_sample = self._n_sample
   }
end

return {
   UAVCoordinates = UAVCoordinates,
}
