require 'cunn'
require 'dp'

model = torch.load('model.t7')
model = model:float()
torch.save('model_converted.t7', model)
