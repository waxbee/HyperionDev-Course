import spacy  # importing spacy

nlp = spacy.load('en_core_web_sm')

# Comparing sentence ===========
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)

print(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


# Comparing words ==========
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Own example comparing words =========
tokens = nlp('crayon pencil stylus paper')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Using en_core_web_md
#
# In the first iteration comparing cat -
# Cat and monkey have the closest similarity as they're both animals
# However, banana has a closer a similarity to the cat than the apple
# In 2nd iteration comparing apple -
# Apple has a closer similarity to the monkey than the cat
# Apple and banana have the closest match because they're both fruits
# In 3rd iteration comparing monkey -
# Monkey has the closest relationship with cat as both animals
# Closely followed by bananas as monkeys eat bananas
# Final iteration comparing banana -
# Banana has the closest relationship with apple as both fruits
# Closely followed by monkeys as they eat bananas
#
# Interesting notes -
# Cats have a higher similarity to bananas than apples, why?
#
# Using en_core_web_sm
#
# In the first iteration comparing cat -
# Cat now most similar to apple and banana instead of monkey
# In 2nd iteration comparing apple -
# apple and monkey now have closer similarity to cat and monkey instead of banana
# In 3rd iteration comparing monkey -
# Monkey and banana now has the lowest similarity and closest to apple
# Final iteration comparing banana -
# banana and monkey have the closest match
#
# Interesting notes - When comparing banana, what causes banana to be most similar to the monkey, but when comparing
# the monkey it is the least similar Why are cat/monkey and banana/apple no longer most similar een though they're
# from the same "group"

