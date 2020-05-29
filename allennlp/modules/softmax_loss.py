import torch
import numpy as np


class SoftmaxLoss(torch.nn.Module):
    """
    Given some embeddings and some targets, applies a linear layer
    to create logits over possible words and then returns the
    negative log likelihood. Note that this class does not add a padding ID.
    """

    def __init__(self, num_words: int, embedding_dim: int) -> None:
        super().__init__()

        # TODO(joelgrus): implement tie_embeddings (maybe)
        self.tie_embeddings = False

        self.softmax_w = torch.nn.Parameter(
            torch.randn(embedding_dim, num_words) / np.sqrt(embedding_dim)
        )
        self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        # Parameters

        embeddings : `torch.Tensor`
            Input tensor with size (batch_size, embedding_dim).
        targets : `torch.Tensor`
            Tensor with the correct class ids with size (batch_size, ).
            Note that it should not include the padding id.
        """

        # Does not do any count normalization / divide by batch size
        probs = torch.nn.functional.log_softmax(
            torch.matmul(embeddings, self.softmax_w) + self.softmax_b, dim=-1
        )

        return torch.nn.functional.nll_loss(probs, targets.long(), reduction="sum")
