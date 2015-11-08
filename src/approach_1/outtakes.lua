-- Load train data (incl. Gaussian normalization)
function load_data(dataset, start_pic_num, pics)

   if opt.dof == 1 then
      label = torch.Tensor(pics)
   else
      label = torch.Tensor(pics, opt.dof)
   end

   -- TODO: think about if a while loop is better here
   i_prime = start_pic_num

   for i = 1, pics do

      local img = image.load(img_folder .. (i_prime - 1) .. ".png")

      if opt.scaleImages then
         img = image.scale(img, img_width, img_height)
      end
   
      true_x = target_x[i_prime]
      true_y = target_y[i_prime]

      if not opt.regression then
	 true_x = math.min(math.floor(true_x),  total_range)
	 true_y = math.min(math.floor(true_y),  total_range)
      end
     
      dataset.data[i] = img

      -- Degrees of freedom
      if opt.dof == 1 then
         label[i] = true_x
      else
         label[i][1] = true_x
      end
      if opt.dof >= 2 then
         label[i][2] = true_y
      end

      i_prime = i_prime + 1

      if opt.regression then
	 dataset.label[i] = true_x
      else
	 dataset.label[i] = makeTargets(true_x, .15)

      end
       
   end

end
