# creating the loss class as a subclass of the nn.Module
import torch
import torch.nn.functional as F

class MyCustomLoss(torch.nn.Module):
    def __init__(self, device, confusion_matrix, mode, weight=None, size_average=True):
      super(MyCustomLoss, self).__init__()
      self._device = device
      self._confusion_matrix = confusion_matrix
      self._mode = mode
      self._fn_cost_matrix = self.generate_fn_cost_matrix(self._confusion_matrix)
      self._fp_cost_matrix = self.generate_fp_cost_matrix(self._confusion_matrix)
      

    def forward(self,outputs, labels):
      # for each mini batch:
      # MISCLASSIFICATION COST AND LOSS
      # device = get_default_device()
        if self._mode == 'fn':
            cost_matrix = self.generate_fn_cost_matrix(self._confusion_matrix)
        elif self._mode == 'fp':
            cost_matrix = self.generate_fp_cost_matrix(self._confusion_matrix)
        
        # if the input to self._mode is not 'fn' or 'fp', then the cost matrix is set to be the identity matrix
        else:
            print('The mode is not recognized. The cost matrix is set to be the identity matrix.')
            dimension = len(self._confusion_matrix)
            cost_matrix = torch.eye(dimension, dimension)

        cost_matrix.to(self._device)
        
        outputs_sm = F.softmax(outputs, dim=1)
        batch_size = outputs.size()[0]
        # select rows from loss matrix based on labels:
        cost_coeff = cost_matrix[labels,:].to(self._device)
        # calculate weighted cost coefficient based on the normalized model outputs i.e., class probability
        # the higher the predicted class probability is concentrated at the correct class column, 
        # the lower the cost coefficient as the corresponding value of the cost matrix is zero
        # during training, back propagation tries to allocate more probability to the correct class to minimize the loss
        weighted_coeff = cost_coeff*outputs_sm

        # suming all the weights and average by batch size for overall effect of the cost matrix
        loss_coeff = torch.sum(weighted_coeff)/batch_size

        # NORMAL CE LOSS
        outputs_log_sm = F.log_softmax(outputs, dim=1)   # normalized with softmax to sum to 1 in the 1st dimension, and apply log
        outputs_true_label = outputs_log_sm[range(batch_size), labels] # pick the values corresponding to the labels => effectively mimicking the delta function
        ce_loss = -torch.sum(outputs_true_label)/batch_size

        # OVERALL LOSS
        custom_loss = loss_coeff*ce_loss

        return custom_loss
    
    def generate_fn_cost_matrix(self, confusion_matrix):
        # set all elements of cost_matrix to zeros:
        dimension = len(confusion_matrix)
        cost_matrix = torch.zeros(dimension, dimension)
        for j in range(dimension):
            for i in range(dimension):
                cost_matrix[i, j] = confusion_matrix[i, j] / \
                    (torch.sum(confusion_matrix[:, j]) - confusion_matrix[j, j])
        # set diagonal back to 0
        for i in range(dimension):
            cost_matrix[i:i+1, i:i+1] = 0
        # multiply all elements with 100
        cost_matrix = cost_matrix*100
        return cost_matrix


    def generate_fp_cost_matrix(self, confusion_matrix):
        # set all elements of cost_matrix to zeros:
        dimension = len(confusion_matrix)
        cost_matrix = torch.zeros(dimension, dimension)
        for i in range(dimension):
            for j in range(dimension):
                cost_matrix[i, j] = confusion_matrix[i, j] / \
                    (torch.sum(confusion_matrix[i, :]) - confusion_matrix[i, i])
        # set diagonal back to 0
        for i in range(dimension):
            cost_matrix[i:i+1, i:i+1] = 0
        # multiply all elements with 100
        cost_matrix = cost_matrix*100
        return cost_matrix