# Import necessary libraries
import numpy as np
import torch
import transformers

# Define the cross-entropy loss function with no reduction, meaning it returns the loss for each element
ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
# Define the softmax function to convert logits to probabilities
softmax_fn = torch.nn.Softmax(dim=-1)

def perplexity_divided_by_entropy(ppl, individual_entropy):
    """
    Compute the ratio of perplexity to entropy for each token in the input.
    
    Parameters:
    ppl (list of list of float): Perplexity values for each token.
    individual_entropy (list of list of float): Entropy values for each token.
    
    Returns:
    divided_values (list of list of float): The ratio of perplexity to entropy for each token.
    """
    divided_values = []
    for ppl_value, entropy_value in zip(ppl, individual_entropy):
        divided_row = [ppl_elem / entropy_elem for ppl_elem, entropy_elem in zip(ppl_value, entropy_value)]
        divided_values.append(divided_row)
    return divided_values

def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    """
    Calculate the perplexity of the model's predictions.

    Parameters:
    encoding (transformers.BatchEncoding): The tokenized inputs.
    logits (torch.Tensor): The model's output logits.
    median (bool): Whether to use the median instead of the mean for calculating perplexity.
    temperature (float): The temperature to scale the logits.

    Returns:
    ppl (numpy.ndarray): The overall perplexity values.
    individual_ppl (numpy.ndarray): The individual perplexity values for each token.
    """
    # Shift the logits and labels by one position to align them for next-token prediction
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        # Calculate cross-entropy loss and mask padding tokens, replacing them with NaN
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        individual_ppl = ce_nan.cpu().float().numpy()
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        # Calculate cross-entropy loss without masking
        ce = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
        individual_ppl = ce.cpu().float().numpy()
        ppl = (ce * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()
    
    print("individual_ppl ", individual_ppl)

    # Calculate the geometric mean of individual perplexities for overall perplexity
    geometric_means = np.exp(np.mean(np.log(individual_ppl), axis=1))
    overall_ppl = np.mean(geometric_means)
    #print("overall_ppl ", overall_ppl)

    return ppl, individual_ppl

def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    """
    Calculate the entropy of the model's predictions.

    Parameters:
    p_logits (torch.Tensor): The logits from the observer model.
    q_logits (torch.Tensor): The logits from the performer model.
    encoding (transformers.BatchEncoding): The tokenized inputs.
    pad_token_id (int): The token ID used for padding.
    median (bool): Whether to use the median instead of the mean for calculating entropy.
    sample_p (bool): Whether to sample from the probability distribution of the observer model.
    temperature (float): The temperature to scale the logits.

    Returns:
    agg_ce (numpy.ndarray): The aggregated cross-entropy values.
    individual_entropy (numpy.ndarray): The individual entropy values for each token.
    """
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    # Apply softmax to the observer model's logits to get probabilities
    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        # Optionally sample from the observer model's probability distribution
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    # Calculate cross-entropy between the performer model's scores and the observer model's probabilities
    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        # Mask padding tokens and calculate the median cross-entropy
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        individual_entropy = ce_nan.cpu().float().numpy()
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        # Calculate the mean cross-entropy without masking
        individual_entropy = ce.cpu().float().numpy()
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    print("individual_entropy ", individual_entropy)
    
    # Calculate the geometric mean of individual entropies for overall entropy
    geometric_means = np.exp(np.mean(np.log(individual_entropy), axis=1))
    overall_pplx = np.mean(geometric_means)
    #print("overall_pplx ", overall_pplx)

    return agg_ce, individual_entropy
