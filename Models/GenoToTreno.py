import torch

class Geno_to_Treno_MAP(torch.nn.Module):
    # Constructor
    def __init__(self, input_size, output_size, model=0):
        super(Geno_to_Treno_MAP, self).__init__()
        if model:
            # hidden layer
            W_eQTL = model.W_eQTL.shape
            W_poly = model.W_poly.shape
            W_fold = model.W_fold.shape

            self.linear_one = torch.nn.Linear(W_eQTL[0]-1, W_eQTL[1])
            self.linear_two = torch.nn.Linear(W_poly[0]-1, W_poly[1])
            self.linear_three = torch.nn.Linear(W_fold[0]-1, W_fold[1])

            # torch.nn.init.zeros_(self.linear_one.weight)
            # torch.nn.init.zeros_(self.linear_two.weight)
            # torch.nn.init.zeros_(self.linear_three.weight)
            # torch.nn.init.zeros_(self.linear_one.bias)
            # torch.nn.init.zeros_(self.linear_two.bias)
            # torch.nn.init.zeros_(self.linear_three.bias)

        else:
            # hidden layer
            self.linear_one = torch.nn.Linear(input_size, int(input_size/0.78*1.2))
            self.linear_two = torch.nn.Linear(int(input_size/0.78*1.2), int((input_size/0.78)*2.4))
            self.linear_three = torch.nn.Linear(int((input_size/0.78)*2.4), output_size)

        # defining layers as attributes
        self.layer_out1 = None
        self.layer_out2 = None
        self.layer_out3 = None

    # prediction function
    def forward(self, x):
        self.layer_out1 = self.linear_one(x)
        self.layer_out1_relu = torch.relu(self.layer_out1)
        self.layer_out2 = self.linear_two(self.layer_out1_relu)
        self.layer_out3 = self.linear_three(self.layer_out2)
        y_pred = torch.sigmoid(self.layer_out3)
        return y_pred

