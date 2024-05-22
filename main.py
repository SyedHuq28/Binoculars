from binoculars import Binoculars

bino = Binoculars()

# ChatGPT (GPT-4) output when prompted with “Can you write a few sentences about a capybara that is an astrophysicist?"
sample_string = '''Dr. Capy Cosmos'''

print(bino.compute_score(sample_string))  # 0.75661373
# print(bino.predict(sample_string))  # 'Most likely AI-Generated'
