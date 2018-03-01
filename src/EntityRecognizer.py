from nltk import DefaultTagger
from nltk.corpus import brown,treebank
import nltk;

class ERecognizer:
    def __init__(self):
        self.grammar="NP: {(<JJ>?<NN>+<NN.*>)|(<JJ>?<NN>+<NNS.*|NNP.*>)|(<JJ>*<NN.*|NNS.*|NNP.*>)|(<NN|NNS|NNP>+<NN.*|NNS.*|NNP.*>)|(<JJ.*>)|(<VB.*|VBZ.*|VBG.*><JJ.*|NN.*|NNS.*|NNP.*>)}"
        self.default_tagger = DefaultTagger('CD')
        self.unigram_tagger='';
        self.bigram_tagger='';
        self.trigram_tagger='';
        self.regexp_tagger = nltk.RegexpTagger([
            (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
            (r'(-|:|;)$', ':'),
            (r'\'*$', 'MD'),
            (r'(The|the|A|a|An|an)$', 'AT'),
            (r'.*able$', 'JJ'),
            (r'^[A-Z].*$', 'NNP'),
            (r'.*ness$', 'NN'),
            (r'.*ly$', 'RB'),
            (r'.*s$', 'NNS'),
            (r'.*ing$', 'VBG'),
            (r'.*ed$', 'VBD'),
            (r'.*', 'NN'),
            ])
        print(self.grammar)
        self.train_taggers()
        print("completed training")
        self.stopper_words=['i','I']

    def extract_entity_names(self,t):
         entity_names = []
         refine_names = []
         if hasattr(t, 'label') and t.label:
             
             if t.label() == 'NP':
                 list_names=[]
                 
                 for child,tag in t:
                     print(child)
                     if child not in self.stopper_words:
                         list_names.append(child)
                     else:
                         print("removing this word")
                         print(child)
                     
                     
                 entity_names.append(' '.join([element for element in list_names]))        
                         
                         
                     
                 #entity_names.append(' '.join([child[0] for child in t if child[0] not in self.stopper_words]))
              
             else:
                 for child in t:
                     entity_names.extend(self.extract_entity_names(child))
         refine_names.extend([phrase for phrase in entity_names if phrase != ''])
         #print(refine_names)
         return refine_names
    def train_taggers(self):
        #using treebank corpus
        tagged_sentences=nltk.corpus.treebank.tagged_sents()
        # let's keep 20% of the data for testing, and 80 for training
        i = int(len(tagged_sentences)*0.2)
        train_sentences = tagged_sentences[i:]
        test_sentences = tagged_sentences[:i]
        self.unigram_tagger = nltk.UnigramTagger(train_sentences, backoff=self.regexp_tagger)
        self.bigram_tagger = nltk.BigramTagger(train_sentences, backoff=self.unigram_tagger)
        self.trigram_tagger = nltk.TrigramTagger(train_sentences, backoff=self.bigram_tagger)

        #test accuracy of the tagger
        accuracy=self.trigram_tagger.evaluate(test_sentences)
        print('accuracy of trigram tagging',accuracy)
        accuracy=self.bigram_tagger.evaluate(test_sentences)
        print('accuracy of biigram tagging',accuracy)
        accuracy=self.unigram_tagger.evaluate(test_sentences)
        print('accuracy of unigram tagging',accuracy)

    def extract_phrases(self,text):
        regexparser = nltk.RegexpParser(self.grammar)
        tri_tagger=self.trigram_tagger
        tags=tri_tagger.tag(nltk.word_tokenize(text))
        result=regexparser.parse(tags)
        return self.extract_entity_names(result);

    def get_pos_tags(self,text):
        tags=self.trigram_tagger.tag(nltk.word_tokenize(text))
        #print(tags)
        return tags
        


        
        
        
        
      
                 
            
            
        
   
    
    
        
            
        
            
                
                
    


        
