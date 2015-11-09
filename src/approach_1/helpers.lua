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

-- Sleep for a specified time in seconds
function sleep(n)
  os.execute("sleep " .. tonumber(n))
end


-- Take a 1D-tensor (e.g. with size 300), and split it into classes
-- For example, 1-30: class 1; 31 - 60: class 2; etc.
function to_classes(predictions, classes) 

   if opt.regression then

      width = 35
      pos = predictions[1]
      pos = normalized_to_raw_num(pos, mean_target, std_target)

--      print("Pos is", pos)
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
