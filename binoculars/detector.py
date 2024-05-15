from typing import Union

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import huggingface_config
from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy, perplexity_divided_by_entropy

torch.set_grad_enabled(False)

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-rw-1b",
                 performer_name_or_path: str = "tiiuae/falcon-rw-1b",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map={"": DEVICE_1},
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   token=huggingface_config["TOKEN"]
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map={"": DEVICE_2},
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    token=huggingface_config["TOKEN"]
                                                                    )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> (transformers.BatchEncoding, list[list[str]]):
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)

        # Convert token IDs back to tokens
        decoded_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in encodings["input_ids"].tolist()]

        # Create a list to store token-divided value pairs
        token_divided_values = []

        print("decoded_tokens ",decoded_tokens)

        return encodings, decoded_tokens, token_divided_values

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits


    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:


        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings, decoded_tokens, token_divided_values = self._tokenize(batch)
        print("encodings ",encodings)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl, individual_ppl = perplexity(encodings, performer_logits)
        print("performer_logits ",performer_logits)
        print("ppl ",ppl)
        x_ppl,individual_entropy = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        print("x_ppl ",x_ppl)
        binoculars_scores = ppl / x_ppl
        print("bscore ",binoculars_scores)
        binoculars_scores = binoculars_scores.tolist()
        divided_values = perplexity_divided_by_entropy(individual_ppl, individual_entropy)

        # Print the result
        print("d values", divided_values)
        flattened_values = np.concatenate(divided_values)
        average_value = np.mean(flattened_values)

        # Associate each token with its divided value
        for tokens, values in zip(decoded_tokens, divided_values):
            token_divided_values.extend(zip(tokens, values))

        #print("avg ", average_value)

        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores, token_divided_values



    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred
