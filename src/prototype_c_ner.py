import nltk
import nltk.tag, nltk.data
import string
from nltk import word_tokenize,pos_tag,ne_chunk
from nltk.corpus import conll2000
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
from nltk.stem.snowball import SnowballStemmer
from nltk import conlltags2tree

default_tagger = nltk.DefaultTagger('NN')

def npchunk_features(sentence, i, history):
     word, pos = sentence[i]
     return {"pos": pos}
class ConsecutiveNPChunkTagger(nltk.TaggerI): 

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train( 
            train_set, algorithm='megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): 
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)






class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        #print(train_data)
        self.unigramTagger = nltk.UnigramTagger(train_data)
        self.bigramTagger = nltk.BigramTagger(train_data,backoff=self.unigramTagger)
        self.tagger = nltk.TrigramTagger(train_data,backoff=self.bigramTagger)

    def parse(self, sentence): 
        pos_tags = [pos for (word,pos) in sentence]
       # print(pos_tags)
        tagged_pos_tags = self.tagger.tag(pos_tags)
       # print(tagged_pos)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
       # print(chunk_tags)
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
       # print(conlltags)
        return nltk.chunk.conlltags2tree(conlltags)

class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
 
        self.feature_detector = features
        #train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                     # for sent in train_sents]
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)
 
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)


def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)
 
    # shift the index with 2, to accommodate the padding
    index += 2
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
 
        'prev-iob': previob,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }


text = """
merchant NN B-CP
banking NN I-CP
in JJ O
New NN B-CP
York NNP I-CP
. . O
  """
trees = nltk.chunk.conllstr2tree(text, chunk_types=["CP"])
print(trees)
#trees.draw()
print('\n')
tagged_sentences = [[((w,t),c) for (w,t,c) in nltk.chunk.tree2conlltags(trees)]]
print(tagged_sentences)
#print(nltk.tree2conlltags(tagged_sents))
print('\n')


train_sentences=tagged_sentences

train_sentences=nltk.corpus.conll2000.chunked_sents('ctrain.txt',chunk_types=['CP','NP','VH'])
print(train_sentences[0])
test_sentences=nltk.corpus.conll2000.chunked_sents('test.txt',chunk_types=['NP'])
testsent="Scooter is my computer Intelligence"



unigramtagger=UnigramChunker(train_sentences)

print(unigramtagger.evaluate(test_sentences))

res=unigramtagger.parse(nltk.pos_tag(nltk.word_tokenize(testsent)))
#iob_triplets = [(w, t, c) for ((w, t), c) in res]

print('---unigram chunker results--')
print(nltk.chunk.tree2conlltags(res))



train_sentences = [[((w,t),c) for (w,t,c) in nltk.chunk.tree2conlltags(sent)]for sent in train_sentences]
test_sentenc = [[((w,t),c) for (w,t,c) in nltk.chunk.tree2conlltags(sent)]for sent in test_sentences]
#print(tag_sentences[0])

chunker = NamedEntityChunker(train_sentences)
result=chunker.parse(nltk.pos_tag(nltk.word_tokenize(testsent)))
#resultmod=chunker.parse()
print('---final named entity chunker---')
print(result)
score=chunker.evaluate(test_sentences)
print(score.accuracy())
#chunker = ConsecutiveNPChunker(train_sentences)
#print(chunker.evaluate(test_sentences))
