# get the llibraries
from sentence_transformers import SentenceTransformer, util 

#  call the model that will vectorize our texts
model = SentenceTransformer("all-MiniLM-L6-v2")

# assign our texts to variables
s1 = "assurance habitation" 
s2 = "contrat d’assurance maison" 
s3 = "voiture électrique"

# encode the senteces
e1, e2, e3 = model.encode([s1, s2, s3])

#  print them to see the result
# print('my first sentece: ')
# print(e1)
# print('----------------------------')
# print('my seconde sentece: ')
# print(e2)
# print('----------------------------')
# print('my third sentece: ')
# print(e3)
# print('----------------------------')

print(util.cos_sim(e1, e2))  # proche 
print(util.cos_sim(e1, e3))  # éloigné