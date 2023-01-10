import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, model_1, model_2):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        # Remove last linear layer from both models:
        self.model_1.fc = nn.Identity()
        self.model_2.fc = nn.Identity()

        # Add a new linear layer:
        # we are adding up the output logits in the forward pass, hence keeping 2048, which is the number of in_attributes to both resnet and inception
        self.fc = nn.Linear(2048, 5) 

    def forward(self, x):
        x1 = self.model_1(x.clone())
        # the output of inceptionv3 is InceptionOutputs class, a tuple with .logits and .auxlogits,
        # taking the logits tensor 
        x2 = self.model_2(x)[0]
        # adding the outputs of the 2 models 
        x = x1 + x2
        # return x with output of the new linear layer (5 classes)
        x = self.fc(x)
        return x