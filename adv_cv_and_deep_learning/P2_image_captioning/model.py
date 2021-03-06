import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)        #resnet = models.resnet50(pretrained=True)
        for param in self.densenet121.parameters():
            param.requires_grad_(False)
        
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, embed_size)
        )

    def forward(self, images):
        features = self.densenet121(images)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        # Discard the <end> word to aboid missamatching 
        captions = captions[:, :-1] 
        self.hidden = self.init_hidden(features.shape[0])
        embed = self.embedding(captions)
        
        # concatenate the features and the caption embeddings
        dec_in = torch.cat((features.unsqueeze(1), embed), dim=1)
        lstm_out, self.hidden = self.lstm(dec_in, self.hidden)
        outputs = self.fc1(lstm_out)
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # Inizialization of the hidden layer
        self.hidden = self.init_hidden(inputs.shape[0])
        caption = []
        
        # loop over the max len caption characters
        for tensor in range(max_len):
            
            lstm_out, self.hidden = self.lstm(inputs, self.hidden)
            outputs = self.fc1(lstm_out.squeeze(1))
            out = outputs.max(dim=1)[1]
            caption.append(out.item())
            print(caption)
            
            inputs = self.embedding(out)
            inputs = inputs.unsqueeze(1)
        return caption

    def init_hidden(self, batch_size):
        return (torch.zeros((1, batch_size, self.hidden_size), device="cuda"), 
                torch.zeros((1, batch_size, self.hidden_size), device="cuda"))