Getting Started
Installation
To run the implementation of Binoculars, you can clone this repository and install the package using pip. This code was developed and tested on Python This code was developed and tested with Python 3.9. To install the package, run the following commands:

$ git clone https://github.com/SyedHuq28/Binoculars_Syed.git
$ cd Binoculars_Syed
$ pip install -e .
Usage
Please note, this implementation comes with a fixed global threshold that is used to classify the input as AI-generated or not. This threshold is selected using Falcon-7B and Falcon-7B-Instruct models for scoring. If you want to use different scoring models, you can pass it as an argument to the Binoculars class. Please read the paper for more details about the Binoculars work.

To detect AI-generated text, please use the following code snippet:

from binoculars import Binoculars

bino = Binoculars()

# ChatGPT (GPT-4) output when prompted with “Can you write a few sentences about a capybara that is an astrophysicist?"
sample_string = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his 
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret 
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he 
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the 
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to 
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

print(bino.compute_score(sample_string))  # 0.75661373
print(bino.predict(sample_string))  # 'Most likely AI-Generated'
In the above code, user can also pass a list of str to compute_score method to get results for the entire batch of samples.
