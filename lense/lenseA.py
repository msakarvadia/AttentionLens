import torch.nn as nn
import torch

class LenseA(nn.Module):
    def __init__(self, n_head, d_model, d_vocab):
        super(LenseA, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.linears = nn.ModuleList([nn.Linear(d_model, d_vocab) for _ in range(n_head)])

    def forward(self, input_tensor):
        batch_size, pos, n_head, d_model = input_tensor.size()

        output_tensors = torch.empty((batch_size, pos, self.n_head, self.d_vocab), device=input_tensor.device)

        for i in range(n_head):
            output_pos = torch.empty((batch_size, pos, self.d_vocab), device=input_tensor.device)

            for j in range(pos):
                # Select the i-th head for the j-th position
                input_pos = input_tensor[:, j, i, :]

                # Reshape the input tensor to [batch_size, d_model]
                input_reshaped = input_pos.reshape(batch_size, d_model)

                # Pass the reshaped input through the linear layer
                output_reshaped = self.linears[i](input_reshaped)

                # print(output_reshaped.requires_grad, output_reshaped.grad_fn)

                output_pos[:, j, :] = output_reshaped

            # Concatenate the output tensors along the second dimension
            # concatenated_output_pos = torch.stack(output_pos, dim=1)

            output_tensors[:, :, i, :] = output_pos

        # Sum across the second to last dimension
        summed_output = output_tensors.sum(dim=2)
        #print(summed_output.size())

        return summed_output

