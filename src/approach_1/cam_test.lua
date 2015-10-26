require 'camera'
require 'image'
require 'torch'
require 'nn'


dev = 1
width = 320
height = 240
fps = 30

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

   frame = cam:forward()  -- return the next frame available
   frame = image.scale(frame, 112, 112)

   print(frame)
   
   image.display{win=win, image=frame}  -- display frame

   -- Define the normalization neighborhood:
   neighborhood = image.gaussian1D(13)
   
   -- Define our local normalization operator (It is an actual nn module, 
   -- which could be inserted into a trainable model):
   normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
   
   -- Normalize all channels locally:
   for c in ipairs(3) do
	 frame[{ {c},{},{} }] = normalization:forward(frame[{ {c},{},{} }])
      end

   
   pred = model:forward(frame)
   max, pos = pred:max(1)

   print(max, pos)

end

cam:stop() -- release the camera
