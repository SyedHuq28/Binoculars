import numpy as np
import torch
import transformers

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def perplexity_divided_by_entropy(ppl, individual_entropy):
    divided_values = []
    for ppl_value, entropy_value in zip(ppl, individual_entropy):
        divided_row = [ppl_elem / entropy_elem for ppl_elem, entropy_elem in zip(ppl_value, entropy_value)]
        divided_values.append(divided_row)
    return divided_values

def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        individual_ppl = np.exp(np.nanmean(ce_nan.cpu().float().numpy(), axis=1))  # Use nanmean instead of nanmedian
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ce = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
        individual_ppl = np.exp(np.mean(ce.cpu().float().numpy(), axis=1))  # Use mean instead of sum
        ppl = np.mean(individual_ppl)
    print("individual_ppl ", individual_ppl)
    print("ppl ", ppl)
    return ppl, individual_ppl

def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        individual_ppl = ce_nan.cpu().float().numpy()
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ce = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
        individual_ppl = ce.cpu().float().numpy()
        ppl = (ce * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()
    print("individual_ppl ",individual_ppl)


    
    
    geometric_means = np.exp(np.mean(np.log(individual_ppl), axis=1))
    # Calculate the overall perplexity by taking the mean of geometric means
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
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        individual_entropy = ce_nan.cpu().float().numpy()
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        individual_entropy = ce.cpu().float().numpy()
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    print("individual_entropy ",individual_entropy) 
    
    geometric_means = np.exp(np.mean(np.log(individual_entropy), axis=1))
    # Calculate the overall perplexity by taking the mean of geometric means
    overall_pplx = np.mean(geometric_means)
    #print("overall_pplx ", overall_pplx)

    
    return agg_ce, individual_entropy

