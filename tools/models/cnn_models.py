import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.models.base_models import BaseModel
    
class CNN_1l(BaseModel):
    def __init__(self, kernel_size, input_size=1):
        super(CNN_1l, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size=kernel_size, padding=0)   
        
        self.total_padding = kernel_size - 1
        self.left_padding = self.total_padding // 2
        self.right_padding = self.total_padding - self.left_padding

    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        
        x = F.pad(x, (self.left_padding, self.right_padding), mode='reflect')
        x = self.conv1(x)
        
        x = x.permute(0, 2, 1)
        
        return x
    
class CNN_2l(BaseModel):
    def __init__(self, kernel_size_1, kernel_size_2, hidden_size, input_size=1):
        super(CNN_2l, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size_1, padding=0)    
        self.conv2 = nn.Conv1d(hidden_size, input_size, kernel_size=kernel_size_2, padding=0)      
        
        self.total_padding_1 = kernel_size_1 - 1
        self.left_padding_1 = self.total_padding_1 // 2
        self.right_padding_1 = self.total_padding_1 - self.left_padding_1
        
        self.total_padding_2 = kernel_size_2 - 1
        self.left_padding_2 = self.total_padding_2 // 2
        self.right_padding_2 = self.total_padding_2 - self.left_padding_2

    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        
        x = F.pad(x, (self.left_padding_1, self.right_padding_1), mode='reflect')
        x = F.relu(self.conv1(x))
        x = F.pad(x, (self.left_padding_2, self.right_padding_2), mode='reflect')
        x = self.conv2(x)
        
        x = x.permute(0, 2, 1)
        
        return x
    
class CNN_3l(BaseModel):
    def __init__(self, kernel_size_1, kernel_size_2, kernel_size_3, hidden_size_1, hidden_size_2, input_size=1):
        super(CNN_3l, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, hidden_size_1, kernel_size=kernel_size_1, padding=0)    
        self.conv2 = nn.Conv1d(hidden_size_1, hidden_size_2, kernel_size=kernel_size_2, padding=0)      
        self.conv3 = nn.Conv1d(hidden_size_2, input_size, kernel_size=kernel_size_3, padding=0)
        
        self.total_padding_1 = kernel_size_1 - 1
        self.left_padding_1 = self.total_padding_1 // 2
        self.right_padding_1 = self.total_padding_1 - self.left_padding_1
        
        self.total_padding_2 = kernel_size_2 - 1
        self.left_padding_2 = self.total_padding_2 // 2
        self.right_padding_2 = self.total_padding_2 - self.left_padding_2

        self.total_padding_3 = kernel_size_3 - 1
        self.left_padding_3 = self.total_padding_3 // 2
        self.right_padding_3 = self.total_padding_3 - self.left_padding_3

    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        
        x = F.pad(x, (self.left_padding_1, self.right_padding_1), mode='reflect')
        x = F.relu(self.conv1(x))
        x = F.pad(x, (self.left_padding_2, self.right_padding_2), mode='reflect')
        x = F.relu(self.conv2(x))
        x = F.pad(x, (self.left_padding_3, self.right_padding_3), mode='reflect')
        x = self.conv3(x)
        
        x = x.permute(0, 2, 1)
        
        return x

class CNN_4l(BaseModel):
    def __init__(
        self, kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4, 
        hidden_size_1, hidden_size_2, hidden_size_3, input_size=1
    ):
        super(CNN_4l, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, hidden_size_1, kernel_size=kernel_size_1, padding=0)    
        self.conv2 = nn.Conv1d(hidden_size_1, hidden_size_2, kernel_size=kernel_size_2, padding=0)      
        self.conv3 = nn.Conv1d(hidden_size_2, hidden_size_3, kernel_size=kernel_size_3, padding=0)
        self.conv4 = nn.Conv1d(hidden_size_3, input_size, kernel_size=kernel_size_4, padding=0)
        
        self.total_padding_1 = kernel_size_1 - 1
        self.left_padding_1 = self.total_padding_1 // 2
        self.right_padding_1 = self.total_padding_1 - self.left_padding_1
        
        self.total_padding_2 = kernel_size_2 - 1
        self.left_padding_2 = self.total_padding_2 // 2
        self.right_padding_2 = self.total_padding_2 - self.left_padding_2

        self.total_padding_3 = kernel_size_3 - 1
        self.left_padding_3 = self.total_padding_3 // 2
        self.right_padding_3 = self.total_padding_3 - self.left_padding_3
        
        self.total_padding_4 = kernel_size_4 - 1
        self.left_padding_4 = self.total_padding_4 // 2
        self.right_padding_4 = self.total_padding_4 - self.left_padding_4

    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        
        x = F.pad(x, (self.left_padding_1, self.right_padding_1), mode='reflect')
        x = F.relu(self.conv1(x))
        x = F.pad(x, (self.left_padding_2, self.right_padding_2), mode='reflect')
        x = F.relu(self.conv2(x))
        x = F.pad(x, (self.left_padding_3, self.right_padding_3), mode='reflect')
        x = F.relu(self.conv3(x))
        x = F.pad(x, (self.left_padding_4, self.right_padding_4), mode='reflect')
        x = self.conv4(x)
        
        x = x.permute(0, 2, 1)
        
        return x

class CNN_4l_Adapter(BaseModel):
    def __init__(self, cnn_4l_model, kernel_size):
        super(CNN_4l_Adapter, self).__init__()
        
        self.conv1 = cnn_4l_model.conv1 
        self.conv2 = cnn_4l_model.conv2
        self.conv3 = cnn_4l_model.conv3
        self.conv4 = cnn_4l_model.conv4
        self.adapter = nn.Conv1d(self.conv3.out_channels, self.conv4.in_channels, kernel_size=kernel_size, padding=0)

        self.total_padding_1 = self.conv1.kernel_size[0] - 1
        self.left_padding_1 = self.total_padding_1 // 2
        self.right_padding_1 = self.total_padding_1 - self.left_padding_1
        
        self.total_padding_2 = self.conv2.kernel_size[0] - 1
        self.left_padding_2 = self.total_padding_2 // 2
        self.right_padding_2 = self.total_padding_2 - self.left_padding_2

        self.total_padding_3 = self.conv3.kernel_size[0] - 1
        self.left_padding_3 = self.total_padding_3 // 2
        self.right_padding_3 = self.total_padding_3 - self.left_padding_3
        
        self.total_padding_4 = self.conv4.kernel_size[0] - 1
        self.left_padding_4 = self.total_padding_4 // 2
        self.right_padding_4 = self.total_padding_4 - self.left_padding_4

        self.total_padding_adapter = kernel_size - 1
        self.left_padding_adapter = self.total_padding_adapter // 2
        self.right_padding_adapter = self.total_padding_adapter - self.left_padding_adapter

        # Freeze original layers
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        
        # Frozen layers
        x = F.pad(x, (self.left_padding_1, self.right_padding_1), mode='reflect')
        x = F.relu(self.conv1(x))
        x = F.pad(x, (self.left_padding_2, self.right_padding_2), mode='reflect')
        x = F.relu(self.conv2(x))
        x = F.pad(x, (self.left_padding_3, self.right_padding_3), mode='reflect')
        x = F.relu(self.conv3(x))

        # Trainable layer
        x = F.pad(x, (self.left_padding_adapter, self.right_padding_adapter), mode='reflect')
        x = F.relu(self.adapter(x))

        # Frozen layer
        x = F.pad(x, (self.left_padding_4, self.right_padding_4), mode='reflect')
        x = self.conv4(x)
        
        x = x.permute(0, 2, 1)
        
        return x

class CNN_4l_Adapter2(BaseModel):
    def __init__(self, cnn_4l_model, kernel_size_1, kernel_size_2):
        super(CNN_4l_Adapter2, self).__init__()
        
        self.conv1 = cnn_4l_model.conv1 
        self.conv2 = cnn_4l_model.conv2
        self.conv3 = cnn_4l_model.conv3
        self.conv4 = cnn_4l_model.conv4
        self.adapter1 = nn.Conv1d(self.conv2.out_channels, self.conv3.in_channels, kernel_size=kernel_size_1, padding=0)
        self.adapter2 = nn.Conv1d(self.conv3.out_channels, self.conv4.in_channels, kernel_size=kernel_size_2, padding=0)

        self.total_padding_1 = self.conv1.kernel_size[0] - 1
        self.left_padding_1 = self.total_padding_1 // 2
        self.right_padding_1 = self.total_padding_1 - self.left_padding_1
        
        self.total_padding_2 = self.conv2.kernel_size[0] - 1
        self.left_padding_2 = self.total_padding_2 // 2
        self.right_padding_2 = self.total_padding_2 - self.left_padding_2

        self.total_padding_3 = self.conv3.kernel_size[0] - 1
        self.left_padding_3 = self.total_padding_3 // 2
        self.right_padding_3 = self.total_padding_3 - self.left_padding_3
        
        self.total_padding_4 = self.conv4.kernel_size[0] - 1
        self.left_padding_4 = self.total_padding_4 // 2
        self.right_padding_4 = self.total_padding_4 - self.left_padding_4

        self.total_padding_adapter1 = kernel_size_1 - 1
        self.left_padding_adapter1 = self.total_padding_adapter1 // 2
        self.right_padding_adapter1 = self.total_padding_adapter1 - self.left_padding_adapter1

        self.total_padding_adapter2 = kernel_size_2 - 1
        self.left_padding_adapter2 = self.total_padding_adapter2 // 2
        self.right_padding_adapter2 = self.total_padding_adapter2 - self.left_padding_adapter2

        # Freeze original layers
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        
        # Frozen layers
        x = F.pad(x, (self.left_padding_1, self.right_padding_1), mode='reflect')
        x = F.relu(self.conv1(x))
        x = F.pad(x, (self.left_padding_2, self.right_padding_2), mode='reflect')
        x = F.relu(self.conv2(x))

        # Trainable layer
        x = F.pad(x, (self.left_padding_adapter1, self.right_padding_adapter1), mode='reflect')
        x = F.relu(self.adapter1(x))

        # Frozen layer
        x = F.pad(x, (self.left_padding_3, self.right_padding_3), mode='reflect')
        x = F.relu(self.conv3(x))

        # Trainable layer
        x = F.pad(x, (self.left_padding_adapter2, self.right_padding_adapter2), mode='reflect')
        x = F.relu(self.adapter2(x))

        # Frozen layer
        x = F.pad(x, (self.left_padding_4, self.right_padding_4), mode='reflect')
        x = self.conv4(x)
        
        x = x.permute(0, 2, 1)
        
        return x

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        