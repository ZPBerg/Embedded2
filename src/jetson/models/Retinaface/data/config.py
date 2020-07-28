# Inference configurations
cfg_inference = {
    'image_shape': (480, 640),
    'resize': 1,                #Set resize factor. Resize factor changes with input shape.
                                #For example, input_shape = (240, 320) => resize = 0.5
    'top_k_before_nms' : 5000,  # Keep top k detections before NMS
    'top_k_after_nms': 750,     #Keep top k detections after NMS
    'nms_thresh': 0.3           #Non-max suppression threshold
}


# MobileNetV1 backbone configurations for training
cfg_mnet = {
    'name': 'mobilenet0.25',                                    #Name of model
    'min_sizes': [[16, 32], [64, 128], [256, 512]],             #Anchor box sizes
    'steps': [8, 16, 32],                                       #Learning rate steps
    'variance': [0.1, 0.2],                                     #Pixel variances
    'clip': False,                                              #Enable gradient clipping
    'loc_weight': 2.0,                                          #localization weights
    'gpu_train': True,                                          #Train on GPU
    'batch_size': 32,                                           #Number of images in 1 batch
    'ngpu': 1,                                                  #Number of GPUs
    'epoch': 250,                                               #Number of epochs
    'decay1': 190,                                              #First Decay amount
    'decay2': 220,                                              #Second Decay amount
    'image_size': 640,                                          #Input image size (for training)
    'pretrain': False,                                          #Use pretrained model
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},   #Layers to train
    'in_channel': 32,                                           #Number of input channel
    'out_channel': 64                                           #Number of output channel
}

# Resnet50 backbone configurations for training
cfg_re50 = {
    'name': 'Resnet50',                                         #Name of model
    'min_sizes': [[16, 32], [64, 128], [256, 512]],             #Anchor box sizes
    'steps': [8, 16, 32],                                       #Learning rate steps
    'variance': [0.1, 0.2],                                     #Pixel variances
    'clip': False,                                              #Enable gradient clipping
    'loc_weight': 2.0,                                          #localization weights
    'gpu_train': True,                                          #Train on GPU
    'batch_size': 24,                                           #Number of images in 1 batch
    'ngpu': 4,                                                  #Number of GPUs
    'epoch': 100,                                               #Number of epochs
    'decay1': 70,                                               #First Decay amount
    'decay2': 90,                                               #Second Decay amount
    'image_size': 840,                                          #Input image size (for training)
    'pretrain': True,                                           #Use pretrained model
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},   #Layers to train
    'in_channel': 256,                                          #Number of input channel
    'out_channel': 256                                          #Number of output channel
}
