**Getting Started** <br>
**Installation** <br>
To run the implementation of Binoculars, you can clone this repository and install the package using pip. This code was developed and tested on Python This code was developed and tested with Python 3.9. To install and run the package, run the following commands:
<br><br>
$ git clone https://github.com/SyedHuq28/Binoculars_Syed.git<br>
$ cd Binoculars_Syed<br>
$ pip install -e .<br>
$ python main.py<br><br>
Every time you make any changes in detector.py or metrics.py, please reinstall the package (pip install -e .) and then run main.py. <br><br>

<b>Description:</b><br>
This function computes a score indicating whether the input text is AI-generated or human-generated.<br><br>

<b>Parameters:</b><br>

input_text (Union[list[str], str]): The input text or a list of input texts to be evaluated.<br><br>
<b>Returns:</b><br>

<b>Steps to Compute the Score:</b><br>

<b>Tokenization and Detokenization:</b><br>
Tokenize and detokenize the input text and prepare encodings for model input.<br><br>
<b>Model Inference:</b><br>
Obtain logits from the observer and performer models for the given encodings.<br><br>
<b>Calculate Perplexity and Entropy:</b><br>
Perplexity and the entropy of the texts are predicted.<br><br>
<b>Calculate Binoculars Scores:</b><br>
Binoculars scores are calculated as the ratio of perplexity to entropy.<br><br>
<b>Calculate Token-Divided Values:</b><br>
Token-Divided Values are calculated using the perplexity-divided-by-entropy function.<br><br>
<b>Return Scores and Token-Divided Values:</b><br>
Return the computed scores and token-divided values (pair of tokens and the numerical value of each token).<br><br>
<b>Output Variables Explained:</b><br>

<b>decoded_tokens</b>: A list of lists where each inner list represents the tokens decoded from the input text.<br>
<b>encodings</b>: A dictionary containing token IDs (input_ids) and attention masks (attention_mask) obtained after tokenization.<br>
<b>individual_ppl</b>: A list of lists representing the individual perplexity values for each token.<br>
<b>performer_logits</b>: Logits obtained from the performer model.<br>
<b>ppl</b>: The overall perplexity value for the input text.<br>
<b>individual_entropy</b>: A list of lists representing the individual entropy values for each token.<br>
<b>x_ppl</b>: The overall entropy value for the input text.<br>
<b>bscore</b>: The computed binoculars score, which is the ratio of perplexity to entropy.<br>
<b>divided_values</b>: A list of tuples where each tuple represents a token and its corresponding divided value (perplexity divided by entropy).<br>
<b>score</b>: The final computed score indicating whether the input text is AI-generated or human-generated.<br>
<b>token_divided_values</b>: A list of tuples where each tuple represents a token and its corresponding divided value (perplexity divided by entropy). <br><br>

**Changing Models in Binoculars**<br>
To change the observer and performer models used in the Binoculars class, you can modify the <b>observer_name_or_path</b> and <b>performer_name_or_path</b> parameters in <b>detector.py</b> file. These parameters specify the model name or path to be loaded.<br><br>

<b>observer_name_or_path</b>="tiiuae/MODEL NAME"<br>
<b>performer_name_or_path</b>="tiiuae/MODEL NAME"<br><br>

Here is the list of models that can be used: <a href="https://huggingface.co/tiiuae">https://huggingface.co/tiiuae</a><br>
Please note, in the original Binoculars repo "Falcon-7B" and "Falcon-7B-Instruct" were used as observer and performer models.
