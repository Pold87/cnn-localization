----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   local batchIdx = 0

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1, testset:size() do

      -- disp progress
      xlua.progress(t, testset:size())


      if opt.batchForward then

         batchIdx = batchIdx + 1
         local batchData = trainset.data:narrow(1, t, 1)
         local batchLabels = trainset.label:narrow(1, t, 1)
         
         -- test sample
         local pred = model:forward(batchData)
         
         confusion:add(to_classes(pred[1], 10), 
                       to_classes(trainset.label[t][1], 10))


      else

         -- get new sample
         local input = testset.data[t]
         if opt.type == 'double' then input = input:double()
         elseif opt.type == 'cuda' then input = input:cuda() end
         local target = testset.label[t]
         
         -- test sample
         local pred = model:forward(input)
         confusion:add(to_classes(pred[1], 10), 
                       to_classes(target[1], 10))
         
      end

   end

   -- timing
   time = sys.clock() - time
   time = time / testset:size()
   print("\n==> time to test 1 sample = " .. (time * 1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end
