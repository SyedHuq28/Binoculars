# Import necessary libraries and modules
from typing import Union
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import custom modules and configurations
from config import huggingface_config
from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy, perplexity_divided_by_entropy

# Disable gradient calculations for performance improvement
torch.set_grad_enabled(False)

# Constants for accuracy and false positive rate thresholds
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low false positive rate

# Set device configuration, preferring GPU if available
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

class Binoculars:
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-rw-1b",
                 performer_name_or_path: str = "tiiuae/falcon-rw-1b",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 2000,
                 mode: str = "low-fpr") -> None:
        """
        Initialize the Binoculars class with models, tokenizer, and configuration.
        """
        # Ensure tokenizer consistency between observer and performer models
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)
        
        # Set mode for threshold values
        self.change_mode(mode)
        
        # Load observer and performer models with specified precision and device
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
        # Set models to evaluation mode
        self.observer_model.eval()
        self.performer_model.eval()

        # Load tokenizer and ensure padding token is set
        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        """
        Change the operating mode of the Binoculars class to adjust thresholds.
        """
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> (transformers.BatchEncoding, list[list[str]]):
        """
        Tokenize input text and prepare encodings for model input.
        """
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

        print("decoded_tokens ", decoded_tokens)

        return encodings, decoded_tokens, token_divided_values

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        """
        Obtain logits from observer and performer models for given encodings.
        """
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        """
        Compute the score indicating whether the input text is AI-generated or human-generated.
        """
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings, decoded_tokens, token_divided_values = self._tokenize(batch)
        print("encodings ", encodings)
        observer_logits, performer_logits = self._get_logits(encodings)
        
        # Calculate perplexity and entropy
        ppl, individual_ppl = perplexity(encodings, performer_logits)
        print("performer_logits ", performer_logits)
        print("ppl ", ppl)
        x_ppl, individual_entropy = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                                            encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        print("x_ppl ", x_ppl)
        
        # Calculate binoculars scores as ratio of perplexity to entropy
        binoculars_scores = ppl / x_ppl
        print("bscore ", binoculars_scores)
        binoculars_scores = binoculars_scores.tolist()
        
        # Calculate values for each token
        divided_values = perplexity_divided_by_entropy(individual_ppl, individual_entropy)
        print("d values", divided_values)
        
        # Flatten and average the values for summary statistics
        flattened_values = np.concatenate(divided_values)
        average_value = np.mean(flattened_values)
        
        # Associate each token with its divided value
        for tokens, values in zip(decoded_tokens, divided_values):
            token_divided_values.extend(zip(tokens, values))

        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores, token_divided_values

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        """
        Predict whether the input text is AI-generated or human-generated based on scores.
        """
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred
