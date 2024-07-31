
from torch.nn import nn

class AdapterHyperNet(nn.Module):
    """This module generates the weights for the meta adapter layers."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterHyperNet, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_task_embeddings = config.train_task_embeddings
        self.task_embedding_dim = config.projected_task_embedding_dim if \
            config.train_task_embeddings else config.task_embedding_dim
        # Considers weight and bias parameters for generating adapter weights.
        self.weight_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim, self.input_dim * self.output_dim))
        self.bias_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim, self.input_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        weight = self.weight_generator(task_embedding).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(task_embedding).view(-1)
        return weight, bias