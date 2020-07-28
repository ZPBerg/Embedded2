import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.utils.box_utils import match, log_sum_exp
from models.RetinaFace.data import cfg_mnet
GPU = cfg_mnet['gpu_train']

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        localization_data, confidence_data, landmarks_data = predictions          #Split prediction tuple into bounding box locations, confidence, facial landmark locations
        num_preds = localization_data.size(0)                           #Get number of predictions
        num_priors = (priors.size(0))                                   #Get number of priors

        # match priors (default boxes) and ground truth boxes
        localization_tensor = torch.Tensor(num_preds, num_priors, 4)        #Create tensor for prior bounding box locations
        landmarks_tensor = torch.Tensor(num_preds, num_priors, 10)     #Create tensor for prior facial landmark locations
        confidence_tensor = torch.LongTensor(num_preds, num_priors)      #Create tensor for prior confidence scores
        for idx in range(num_preds):
            truths = targets[idx][:, :4].data               #Get ground truth data for bound box locations
            labels = targets[idx][:, -1].data               #Get ground truth data for classification label
            landmarks = targets[idx][:, 4:14].data             #Get ground truth data for facial landmark locations
            defaults = priors.data                          #Get priors data
            match(self.threshold, truths, defaults, self.variance, labels, landmarks, localization_tensor, confidence_tensor, landmarks_tensor, idx)   #Match priors to ground truth boxes
        if GPU:
            #If GPU is available, assign priors to perform computation using cuda. (Enable cuda)
            localization_tensor = localization_tensor.cuda()
            confidence_tensor = confidence_tensor.cuda()
            landmarks_tensor = landmarks_tensor.cuda()

        zeros = torch.tensor(0).cuda()                                      #Create a tensor with value 0
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = confidence_tensor > zeros                                               #Get all the priors where confidence is greater than 0
        num_pos_landm = pos1.long().sum(1, keepdim=True)                    #Add 1 to the priors that are greater than 0
        N1 = max(num_pos_landm.data.sum().float(), 1)                       #Get the denominator (normalization factor) to normalize the total loss of landmark locations
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landmarks_data)         #Expand pos_idx1 to the size of landmarks_data
        landmarks_position = landmarks_data[pos_idx1].view(-1, 10)                         #Creates a tensor that shares the same data as the desired indexes of landmarks_data
        landmarks_tensor = landmarks_tensor[pos_idx1].view(-1, 10)                            #Creates a tensor that shares the same data as the desired indexes of landmarks_tensor
        landmarks_loss = F.smooth_l1_loss(landmarks_position, landmarks_tensor, reduction='sum')#Calculate L1 loss


        pos = confidence_tensor != zeros                                               #Get all the priors where confidence is not 0
        confidence_tensor[pos] = 1                                                     #Set the confidence values of these to 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(localization_data)                          #Expand pos_idx to the size of localization_data
        localization_position = localization_data[pos_idx].view(-1, 4)                                           #Creates a tensor that shares the same data as the desired indexes of localization_position
        localization_tensor = localization_tensor[pos_idx].view(-1, 4)                                              #Creates a tensor that shares the same data as the desired indexes of localization_tensor
        localization_loss = F.smooth_l1_loss(localization_position, localization_tensor, reduction='sum')             #Calculate L1 loss

        # Compute max conf across batch for hard negative mining
        batch_conf = confidence_data.view(-1, self.num_classes)                                             #Creates a tensor that shares the same data as confidence_data
        classification_loss = log_sum_exp(batch_conf) - batch_conf.gather(1, confidence_tensor.view(-1, 1))      #Calculate classification loss

        # Hard Negative Mining : Create negative examples where detector previously falsely detects object
        classification_loss[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        classification_loss = classification_loss.view(num_preds, -1)
        _, loss_idx = classification_loss.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(confidence_data)                                 #Expand pos_idx to the shape of confidence_data
        neg_idx = neg.unsqueeze(2).expand_as(confidence_data)                                 #Expand neg_idx to the shape of confidence_data
        conf_p = confidence_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)           #Creates a tensor that shares the same data indexes of confidence_data where the values are greater than 0
        targets_weighted = confidence_tensor[(pos+neg).gt(0)]
        classification_loss = F.cross_entropy(conf_p, targets_weighted, reduction='sum')             #Calculate cross_entropy loss

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)                                  #Normalization factor used for localization loss(bounding box) and classification loss
        localization_loss /= N                                                  #Calculate final localization loss
        classification_loss /= N                                                #Calculate final classification loss
        landmarks_loss /= N1                                                    #Calculate localization loss of facial coordinates

        return localization_loss, classification_loss, landmarks_loss           #Return losses for weight update
