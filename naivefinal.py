import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize 
 
 
 
pos_tweets = [('I love this car', 'positive'), 
               ('This view is amazing', 'positive'), 
               ('I feel great this morning', 'positive'), 
               ('I am so excited about the concert', 'positive'), 
               ('He is my best friend', 'positive')]  
neg_tweets = [('I do not like this car', 'negative'), 
               ('This view is horrible', 'negative'), 
               ('I feel tired this morning', 'negative'), 
               ('I am not looking forward to the concert', 'negative'), 
               ('He is my enemy', 'negative')] 
neut_tweets= [('tajmahal is the seventh wonder of  world', 'neutral'), 
               ('this is my house', 'neutral'), 
               ('i just upgraded my python version', 'neutral'), 
               ('i am using windows', 'neutral'), 
               ('i rushed towards the store today', 'neutral'),
               ('What iPhone hacking case means tech industry law enforcement', 'neutral'), 
               ('i just upgraded my python version', 'neutral'), 
               ('iPhone SE Name new smaller Apple handset revealed The Independent', 'neutral'), 
               ('iPhoneNews reason behind foxconnoficial 62 billion acquisition Sharp Read', 'neutral'),
               ('All latest Brighton amp Hove Albion news iPhone iPad','neutral')]  



            
 
count_pos=0
count_neg=0
count_neut=0
tweets = [] 
for (words, sentiment) in pos_tweets + neg_tweets+neut_tweets: 
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    tweets.append((words_filtered, sentiment)) 
 

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(tweets)) 
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


training_set = nltk.classify.apply_features(extract_features, tweets) 

 
classifier = nltk.NaiveBayesClassifier.train(training_set) 

f=open('rem_spaceFi1.txt', 'r')
for line in f:
  tweet = line
  test=classifier.classify(extract_features(tweet.split()))
  if test=='positive':
   count_pos=count_pos+1
  elif test=='negative':
   count_neg=count_neg+1
  elif test=='neutral':
   count_neut=count_neut+1

pos=count_pos
neg=count_neg
neut=count_neut
print 'count_pos'
print pos
print 'count_neg'
print neg
print 'count_neut'
print neut

total=count_pos+count_neg+count_neut
print total
percentage_pos=pos*100/total
percentage_neg=neg*100/total
percentage_neut=neut*100/total

print 'percentage_pos'
print percentage_pos

print 'percentage_neg'
print percentage_neg

print 'percentage_neut'
print percentage_neut

import matplotlib.pyplot as plt
 
# Data to plot
labels = 'Negative %','positive %','neutral %'
sizes = [percentage_pos, 45.8, 39]
colors = ['red', 'green','lightskyblue']
explode = (0.1, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
