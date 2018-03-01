import nltk
import nltk.tag, nltk.data
from nltk import word_tokenize,pos_tag,ne_chunk
from nltk.corpus import conll2000
default_tagger = nltk.DefaultTagger('NN')
class Chunker(nltk.ChunkParserI):
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
tagged_sents = [[((w,t),c) for (w,t,c) in nltk.chunk.tree2conlltags(trees)]]
print(tagged_sents)
#print(nltk.tree2conlltags(tagged_sents))
print('\n')


train_sentences=tagged_sents

train_sentences=nltk.corpus.conll2000.chunked_sents('ctrain.txt',chunk_types=['CP','VH'])
print(train_sentences[0])
test_sentences=nltk.corpus.conll2000.chunked_sents('ctest.txt',chunk_types=['CP','VH'])
testsent="Bike is  a super fast computer Intelligence"


tagger=Chunker(train_sentences)

print(tagger.evaluate(test_sentences))

res=tagger.parse(nltk.pos_tag(nltk.word_tokenize(testsent)))
#iob_triplets = [(w, t, c) for ((w, t), c) in res]


print(nltk.chunk.tree2conlltags(res))



