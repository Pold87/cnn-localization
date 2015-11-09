require 'dp'

------------------------------------------------------------------------
--[[ ConfusionFeedback ]]--
-- Feedback
-- Adapter for optim.ConfusionMatrix
-- requires 'optim' package
------------------------------------------------------------------------
local ConfusionFeedback, parent = torch.class("dp.ConfusionFeedback", "dp.Feedback")
ConfusionFeedback.isConfusionFeedback = true
   
function ConfusionFeedback:__init(config)
   require 'optim'
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, name = xlua.unpack(
      {config},
      'ConfusionFeedback', 
      'Adapter for optim.ConfusionMatrix',
      {arg='name', type='string', default='confusionfeedback',
       help='name identifying Feedback in reports'}
   )
   config.name = name
   self._output_module = output_module or nn.Identity()
   parent.__init(self, config)
end

function ConfusionFeedback:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function ConfusionFeedback:doneEpoch(report)
   if self._cm and self._verbose then
      print(self._id:toString().." accuracy = "..self._cm.totalValid)
   end
end

function ConfusionFeedback:to_classes(predictions, classes)

   local width = 35

   for i = 1, predictions:size(1) do
      
      class = math.max(math.ceil(predictions[i] / width), 1)
      predictions[i] = math.min(class, classes)
      
   end

end

function ConfusionFeedback:_add(batch, output, report)
   if self._output_module then
      output = self._output_module:updateOutput(output)
   end
   
   if not self._cm then
--      self._cm = optim.ConfusionMatrix(batch:targets():classes())
      self._cm = optim.ConfusionMatrix(10)
      self._cm:zero()
   end
   
   local act = output:view(output:size(1), -1)
   local tgt = batch:targets():forward('b')
      
   if not (torch.isTypeOf(act,'torch.FloatTensor') or torch.isTypeOf(act, 'torch.DoubleTensor')) then
      self._actf = self.actf or torch.FloatTensor()
      self._actf:resize(act:size()):copy(act)
      act = self._actf
   end

   act:mul(st._std[1][1] + st._std_eps)
   tgt:mul(st._std[1][1] + st._std_eps)

   act:add(st._mean[1][1])
   tgt:add(st._mean[1][1])

--   print("act", act)
--   print("tgt", tgt)

   self:to_classes(act[{{}, 1}], 10)
   self:to_classes(tgt, 10)

--   print("act", act)
--   print("tgt", tgt)

   self._cm:batchAdd(act, tgt)
end

function ConfusionFeedback:_reset()
   if self._cm then
      self._cm:zero()
   end
end

function ConfusionFeedback:report()
   local cm = self._cm or {}

   print(self._cm)

   if self._cm then
      cm:updateValids()
   end
   --valid means accuracy
   --union means divide valid classification by sum of rows and cols
   -- (as opposed to just cols.) minus valid classificaiton 
   -- (which is included in each sum)
   return { 
      [self:name()] = {
         matrix = cm.mat,
         per_class = { 
            accuracy = cm.valids,
            union_accuracy = cm.unionvalids,
            avg = {
               accuracy = cm.averageValid,
               union_accuracy = cm.averageUnionValid
            }
         },
         accuracy = cm.totalValid,
         avg_per_class_accuracy = cm.averageValid,
         classes = cm.classes
      },
      n_sample = self._n_sample
   }
end
