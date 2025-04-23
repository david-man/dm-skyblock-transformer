import torch
from torch import nn
import math

class Gate(nn.Module):
    '''
    A Gating Network that takes in J indices about |S| stocks
    '''
    def __init__(self, j, s):
        super(Gate, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(j, s),
            nn.Softmax(dim = -1)
        )
    def forward(self, x):
        return self.model(x)

class Embedder(nn.Module):
    '''
    An unbiased embedding layer that, given K features per |S| stocks, embeds all of them into E embeddings
    '''
    def __init__(self, k, e):
        super(Embedder, self).__init__()
        self.model = nn.Linear(k, e, bias = False)
    def forward(self, x):
        return self.model(x)
class PositionalEncoding(nn.Module):
    '''
    A layer that generates positional encodings for all elements in a |S| x t x E matrix of stocks.
    '''
    def __init__(self, t, e):
        super(PositionalEncoding, self).__init__()
        positional_encoding = torch.zeros(t, e) + torch.arange(0, t).unsqueeze(1) 

        #[[0, 0, 0...], <-- this is the embeddings of one stock at one timestep; thus, we need to positional encode these
        #[1, 1, 1...]]...   
        #This basically numbers each row(each timestep) so that each row is positionally encoded relative to each other

        two_i_term = torch.floor(torch.arange(e) / 2) * 2
        #[0, 0, 2, 2, 4, 4...]. Gets the 2i term in 10000^(2i/d) by

        divide_term = torch.pow(torch.ones(e) * 100004, two_i_term/e)
        #Computes the 10000^(2i/d)
        positional_encoding = positional_encoding / divide_term
        #Now we have n/10000^(2i/d)

        sin_section = positional_encoding[:, 0::2]#even columns
        cos_section = positional_encoding[:, 1::2]#odd columns

        positional_encoding[:, 0::2] = torch.sin(sin_section)
        positional_encoding[:, 1::2] = torch.cos(cos_section)

        #we don't want to apply gradients, so this is a buffer
        self.register_buffer("positional_encoding", positional_encoding)
    
    def forward(self, x):
        return self.positional_encoding
    
class LayerNormalization(nn.Module):
    '''
    Normalizes the final layer via the LayerNorm algorithm
    '''
    def __init__(self, dimension_of_last_layer : int, epsilon = 1e-10):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.Tensor(dimension_of_last_layer))
        self.beta = nn.Parameter(torch.Tensor(dimension_of_last_layer))
    def forward(self, x : torch.Tensor):
        mean = x.mean(dim = -1, keepdim=True)
        variance = x.var(dim = -1, keepdim=True, correction = 0)
        return (x - mean)/torch.sqrt(variance + self.epsilon) * self.gamma + self.beta

class AttentionHead(nn.Module):
    '''
    Single Encoder Attention Head for Transformers. Takes in an input |S| x t x e.
    '''
    def __init__(self, input_dimension, output_dimension):
        super(AttentionHead, self).__init__()
        self.output_dimension = output_dimension
        self.query = nn.Linear(input_dimension, output_dimension, bias = False)
        self.key = nn.Linear(input_dimension, output_dimension, bias = False)
        self.value = nn.Linear(input_dimension, output_dimension, bias = False)
    def forward(self, x):
        qX = self.query(x)#|S| x t x o
        kX = self.key(x)#|S| x t x o
        vX = self.value(x)#|S| x t x o
        
        transposed_kX = torch.transpose(kX, -1, -2)#|S| x o x t
        weights = torch.softmax(torch.matmul(qX, transposed_kX), dim = -1)#|S| x t x t
        resultant = torch.matmul(weights, vX) / math.sqrt(self.output_dimension) #|S| x t x o. contains a scaling factor

        return resultant

class FFN(nn.Module):
    '''
    FeedForward Network from e -> e followed by a dropout. Used in the MHA
    '''
    def __init__(self, input_dimension):
        super(FFN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dimension, 4 * input_dimension),
            nn.ReLU(),
            nn.Linear(4 * input_dimension, input_dimension),
            nn.Dropout()
        )
    def forward(self, x):
        return self.model(x)
class IntraStockMultiheadAttention(nn.Module):
    '''
    MultiAttention Head for IntraStock Aggregation. Takes in an input |S| x t x e that has been layer-normalized
    '''
    def __init__(self, encodings, heads = 1):
        super(IntraStockMultiheadAttention, self).__init__()
        self.heads = heads
        self.head_sizes = []
        for _ in range(heads):#create as close as possible attention head sizes
            self.head_sizes.append(int(encodings / heads))
        self.head_sizes[-1] += encodings - (int(encodings / heads) * heads)
        self.attention_heads = []
        for head_size in self.head_sizes:
            self.attention_heads.append(AttentionHead(encodings, head_size))
        self.layer_norm = LayerNormalization(encodings)
        self.ffn = FFN(encodings)
    def forward(self, x):
        outputs = []
        for i in range(self.heads):
            outputs.append(self.attention_heads[i](x))
        output = torch.concat(outputs, dim = -1)#smash them back together
        residual_output = output + x#considers the residuals
        return residual_output + self.ffn(self.layer_norm(residual_output))
    
class InterStockMultiheadAttention(nn.Module):
    '''
    MultiAttention Head for InterStock Aggregation. Takes in an input |S| x t x e from the IntraStock Aggregation
    '''
    def __init__(self, encodings, heads = 1):
        super(InterStockMultiheadAttention, self).__init__()
        self.heads = heads
        self.head_sizes = []
        for _ in range(heads):#create as close as possible attention head sizes
            self.head_sizes.append(int(encodings / heads))
        self.head_sizes[-1] += encodings - (int(encodings / heads) * heads)
        self.attention_heads = []
        for head_size in self.head_sizes:
            self.attention_heads.append(AttentionHead(encodings, head_size))
        self.layer_norm_1 = LayerNormalization(encodings)
        self.layer_norm_2 = LayerNormalization(encodings)
        self.ffn = FFN(encodings)
    def forward(self, x):
        xT = torch.transpose(x, -2, -3)#t x |S| x e
        xT = self.layer_norm_1(xT)
        outputs = []
        for i in range(self.heads):
            outputs.append(self.attention_heads[i](xT))
        output = torch.concat(outputs, dim = -1)#smash them back together
        output = torch.transpose(output, -2, -3) #retranspose the outputs so its back to #|S| x t x e
        residual_output = output + x#considers the residuals
        return residual_output + self.ffn(self.layer_norm_2(residual_output))

class TemporalAggregation(nn.Module):
    '''
    Temporal Aggregation. Takes in an input |S| x t x e from the InterStock Aggregation
    '''
    def __init__(self, encodings):
        super(TemporalAggregation, self).__init__()
        self.linear = nn.Linear(encodings, encodings, bias = False)
    def forward(self, x):
        x_1 = self.linear(x)#|S| x t x e
        current_timestep = x_1[..., -1, :]#|S| x e
        current_timestep = current_timestep.unsqueeze(dim = -1)#|S| x e x 1
        lam = torch.matmul(x_1, current_timestep)#|S| x 1 x e
        lam = torch.squeeze(lam)#|S| x e
        lam = torch.softmax(lam, dim = -1).unsqueeze(dim = -2)#|S| x 1 x e
        output = torch.matmul(lam, x)#|S| x 1 x e
        return output.squeeze()#|S| x e
    
class Predictor(nn.Module):
    '''
    Final prediction layer. Takes in an input |S| x e from the Temporal Aggregator
    '''
    def __init__(self, encodings):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(encodings, 1, bias = True)
    def forward(self, x):
        return self.linear(x)
class MASTER(nn.Module):
    def __init__(self, 
                 timesteps : int, #what is your t
                 features : int, #initial features to embed into encodings. may literally be 1, but thats okay. these should be the first things in the input
                 gate_inputs : int, #gate input size
                 encodings : int, #encodings to use
                 heads : int = 1,
    ):
        super(MASTER, self).__init__()
        self.features = features
        self.gate_inputs = gate_inputs
        self.gate = Gate(gate_inputs, features)
        self.embedder = Embedder(features, encodings)
        self.position_encoder = PositionalEncoding(timesteps, encodings)
        self.layer_norm = LayerNormalization(encodings)
        self.intrastock = IntraStockMultiheadAttention(encodings, heads)
        self.interstock = InterStockMultiheadAttention(encodings, heads)
        self.temporal_attention = TemporalAggregation(encodings)
        self.predictor = Predictor(encodings)
    
    def forward(self, x):
        #x is of shape B x |S| x t x (f + g)
        x_f = x[..., :self.features]
        if(self.gate_inputs != 0):
            if(self.features == 1):#torch is annoying with the auto-dimensionality reduction
                x_f.unsqueeze(-1)
            x_g = x[..., -1, self.features:]
            if(self.gate_inputs == 1):
                x_g.unsqueeze(-1)
            
            gated_output = self.gate(x_g).unsqueeze(dim = -2)
            x_f = gated_output * x_f
        embedded_input = self.embedder(x_f)#B x |S| x t x e
        positionally_embedded_input = self.position_encoder(embedded_input)#B x |S| x t x e
        residual_input = embedded_input + positionally_embedded_input
        layer_normed_input = self.layer_norm(residual_input)#B x |S| x t x e
        intrastock_input = self.intrastock(layer_normed_input)#B x |S| x t x e
        interstock_input = self.interstock(intrastock_input)#B x |S| x t x e
        temporally_attended_input = self.temporal_attention(interstock_input)#B x |S| x t x e
        outputs = self.predictor(temporally_attended_input)#B x |S| x t x e
        return outputs

if __name__ == '__main__':
    model = MASTER(1, 2, 1, 4)
    k = torch.rand((1,6,5,3))
    print(model(k))
    
