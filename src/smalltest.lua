require 'torch'
require 'math'

function makeTargets(y, stdv)
   -- y : (batch_size, num_keypoints*2)
   -- Y : (batch_size, num_keypoints*2, 98)
   Y = torch.FloatTensor(y:size(1), y:size(2), 98):zero()
   pixels = torch.range(1,98):float()
   local k = 0
   for i=1,y:size(1) do
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
