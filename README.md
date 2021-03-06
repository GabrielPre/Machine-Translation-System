# Deep Neural Network Models for Advanced NLP Problems
Made by Gabriel PRECIGOUT and Stanislas KIESGEN DE RICHTER

## Task 1 : Creating a Machine Translation System

### Introduction

In this notebook, we built a deep neural network that uses Sequence-to-sequence(Seq2seq)


```python
from google.colab import drive
drive.mount('/content/drive/')
```

    Mounted at /content/drive/



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import datetime, os
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from numpy import array, asarray, zeros

%load_ext tensorboard
```

### Definition of the Hyperparameters


```python
BATCH_SIZE = 64
EPOCHS = 25
LSTM_NODES =256
NUM_SENTENCES = 8000
MAX_SENTENCE_LENGTH = 50
MAX_NUM_WORDS = 20000
EMBEDDING_SIZE = 100

```

### Data preprocessing

Randomizing the text file to improve our model 


```python
#with open('/content/drive/MyDrive/Data/fra.txt','r') as source:
#    data = [ (random.random(), line) for line in source ]
#data.sort()
#with open('/content/drive/MyDrive/Data/fra_random.txt','w') as target:
#    for _, line in data:
#        target.write( line )
```

Transforming the data from the text file to an array


```python
input_sentences = []
end_of_sentences_outputs = []
start_of_sentence_outputs = []
test_outputs = []
 
count = 0
for line in open(r'/content/drive/MyDrive/Data/fra_random.txt', encoding="utf-8"):
    count += 1
 
    if count > NUM_SENTENCES:
        break
 
    if '\t' not in line:
        continue
    
    input_sentence, output, attribution = line.rstrip().split('\t')

    test_output = output
    end_of_sentences_output = output + ' <eos>'
    start_of_sentence_output = '<sos> ' + output

    test_outputs.append(test_output)
    input_sentences.append(input_sentence)
    end_of_sentences_outputs.append(end_of_sentences_output)
    start_of_sentence_outputs.append(start_of_sentence_output)
 
print("number of samples in the input_sentences:", len(input_sentences))
print("number of samples in the end_of_sentences_outputs:", len(end_of_sentences_outputs))
print("number of samples in the start-of-sentences_outputs:", len(start_of_sentence_outputs))
```

    number of samples in the input_sentences: 8000
    number of samples in the end_of_sentences_outputs: 8000
    number of samples in the start-of-sentences_outputs: 8000



```python
print(input_sentences)
```

    ['Tom renamed the folder.', "Tom says he hasn't eaten in three days.", 'He is able to speak five languages.', "I'm pretty sure that Tom doesn't have a brother.", "I don't mind a bit.", "It's not as easy as people think.", 'I told you it was dangerous.', 'They tried to discourage him from going.', 'It makes sense.', 'That man looks familiar.', 'What else can go wrong?', 'Tom wanted one, but he had no idea where he could get one.', 'He is a doctor.', 'Tell us a ghost story.', 'What made you ask that question?', "Don't release that dog.", 'You need to work together.', "I can't eat that.", 'Here comes the train.', "I don't want to be a burden to you.", 'The train arrives at platform number 5.', "I don't talk about you behind your back.", "Tom parked the car behind Mary's house.", 
    
    {...}
    
    'Look at me when I talk to you!', 'Can I get you something to drink?', 'Did you recognize any of those people?', 'I had to see you again.', 'You deserve more than that.', 'Come sit with us.', 'Tom {looked very concerned.']


Printing what we're supposed to translate and how it looks


```python
print(input_sentences[42])
print(end_of_sentences_outputs[42])
print(start_of_sentence_outputs[42])
```

    I'm done fooling around now.
    J'en ai fini de faire l'andouille. <eos>
    <sos> J'en ai fini de faire l'andouille.


### Tokenization


```python
input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(input_sentences)
input_integer_sequence = input_tokenizer.texts_to_sequences(input_sentences)

inputs_convert_word_to_wordIndex = input_tokenizer.word_index
print('Total unique words in the input: %s' % len(inputs_convert_word_to_wordIndex))

len_longest_input_sentence = max(len(sen) for sen in input_integer_sequence)
print("Length of longest sentence in input: %g" % len_longest_input_sentence)
```

    Total unique words in the input: 4261
    Length of longest sentence in input: 25



```python
output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(end_of_sentences_outputs + start_of_sentence_outputs)
output_integer_seq = output_tokenizer.texts_to_sequences(end_of_sentences_outputs)
output_input_integer_sequence = output_tokenizer.texts_to_sequences(start_of_sentence_outputs)

word2idx_outputs = output_tokenizer.word_index
print('Total unique words in the output: %s' % len(word2idx_outputs))

num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)
print("Length of longest sentence in the output: %g" % max_out_len)
```

    Total unique words in the output: 8692
    Length of longest sentence in the output: 30


Padding the input_sequence_encoder so that all sentences have the same length


```python
input_sequences_encoder = pad_sequences(input_integer_sequence, maxlen=len_longest_input_sentence)
print("input_sequences_encoder.shape:", input_sequences_encoder.shape)
print("input_sequences_encoder[42]:", input_sequences_encoder[42])
```

    input_sequences_encoder.shape: (8000, 25)
    input_sequences_encoder[42]: [   0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0   21  147 2223  511   95]


Padding the input_sequence_decoder


```python
input_sequences_decoder = pad_sequences(output_input_integer_sequence, maxlen=max_out_len, padding='post')
print("input_sequences_decoder.shape:", input_sequences_decoder.shape)
print("input_sequences_decoder[172]:", input_sequences_decoder[172])
```

    input_sequences_decoder.shape: (8000, 30)
    input_sequences_decoder[172]: [   2   14 2052   70 1464    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0]


### Word Embedding


```python
embeddings_dictionary = dict()

glove = open(r'/content/drive/MyDrive/Data/glove.6B.100d.txt', encoding="utf8")

for line in glove:
    glove_records = line.split()
    glove_word = glove_records[0]
    vector_dimensions = asarray(glove_records[1:], dtype='float32')
    embeddings_dictionary[glove_word] = vector_dimensions
glove.close()
```


```python
num_words = min(MAX_NUM_WORDS, len(inputs_convert_word_to_wordIndex) + 1)
embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
for word, index in inputs_convert_word_to_wordIndex.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
```


```python
embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=len_longest_input_sentence)
```


```python
one_hot_decoder = np.zeros((
        len(input_sentences),
        max_out_len,
        num_words_output
    ),
    dtype='float32'
)
```


```python
one_hot_decoder.shape
```




    (8000, 30, 8693)




```python
output_sequences_decoder = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')
```


```python
for i, d in enumerate(output_sequences_decoder):
    for t, word in enumerate(d):
        one_hot_decoder[i, t, word] = 1
```


```python
inputs_placeholder_encoder = Input(shape=(len_longest_input_sentence,))
x = embedding_layer(inputs_placeholder_encoder)
encoder = LSTM(LSTM_NODES, return_state=True)

encoder_outputs, hidden, cell = encoder(x)
encoder_states = [hidden, cell]
```


```python
inputs_placeholder_decoder = Input(shape=(max_out_len,))

embedding_decoder = Embedding(num_words_output, LSTM_NODES)
inputs_decoder = embedding_decoder(inputs_placeholder_decoder)

lstm_decoder = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = lstm_decoder(inputs_decoder, initial_state=encoder_states)
```


```python
dense_decoder = Dense(num_words_output, activation='softmax')
decoder_outputs = dense_decoder(decoder_outputs)
```


```python
model = Model([inputs_placeholder_encoder,
  inputs_placeholder_decoder], decoder_outputs)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```


```python
plot_model(model, to_file='model_plot_1.png', show_shapes=True, show_layer_names=True)
```




    
![png](output_32_0.png)
    




```python
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
```


```python
model.load_weights("/content/drive/MyDrive/Data/saved_weights.hdf5")
r = model.fit(
    [input_sequences_encoder, input_sequences_decoder],
    one_hot_decoder,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[tensorboard_callback]
)
model.save_weights('/content/drive/MyDrive/Data/saved_weights.hdf5',overwrite=True)
```


```python
%tensorboard --logdir logs
```


    <IPython.core.display.Javascript object>



```python
# Ploting Loss per Epochs
plt.plot(range(len(model.history.history['val_loss'])),model.history.history['loss'])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
```


```python
# Ploting accuracy per Epochs
plt.plot(range(len(model.history.history['val_accuracy'])),model.history.history['accuracy'])
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()
```


    
![png](output_37_0.png)
    



```python
model.save_weights('/content/drive/MyDrive/Data/saved_weights.hdf5',overwrite=True)
```


```python
encoder_model = Model(inputs_placeholder_encoder, encoder_states)
```


```python
decoder_state_input_hidden = Input(shape=(LSTM_NODES,))
decoder_state_input_cell = Input(shape=(LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
```


```python
single_inputs_decoder = Input(shape=(1,))
single_x_inputs_decoder = embedding_decoder(single_inputs_decoder)
```


```python
decoder_outputs, hidden, cell = lstm_decoder(single_x_inputs_decoder, initial_state=decoder_states_inputs)
```


```python
decoder_states = [hidden, cell]
decoder_outputs = dense_decoder(decoder_outputs)
```


```python
decoder_model = Model(
    [single_inputs_decoder] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)
```


```python
plot_model(decoder_model, to_file='model_plot_decoder.png', show_shapes=True, show_layer_names=True)
```




    
![png](output_45_0.png)
    




```python
index_to_word_input = {v:k for k, v in inputs_convert_word_to_wordIndex.items()}
index_to_word_output = {v:k for k, v in word2idx_outputs.items()}
```


```python
def machine_Translation_System(input_sequence):
    values_of_states = encoder_model.predict(input_sequence)
    sequence_translation = np.zeros((1, 1))
    sequence_translation[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, hidden, cell = decoder_model.predict([sequence_translation] + values_of_states)
        index_mts = np.argmax(output_tokens[0, 0, :])

        if eos == index_mts:
            break

        word = ''

        if index_mts > 0:
            word = index_to_word_output[index_mts]
            output_sentence.append(word)

        sequence_translation[0, 0] = index_mts
        values_of_states = [hidden, cell]

    return ' '.join(output_sentence)
```


```python
test_outputs = []
 
count1 = 0
for line1 in open(r'/content/drive/MyDrive/Data/fra_random.txt', encoding="utf-8"):
    count1 += 1
 
    if count1 > 8500:
        break
 
    if '\t' not in line1:
        continue
    
    input_sentence1, output1, attribution1 = line1.rstrip().split('\t')

    test_outputs.append(output1)
test_outputs1 = test_outputs[8000:]
```


```python
print(test_outputs1[2])
```

    Par temps chaud, la sueur permet ?? l'homme de r??guler la temp??rature de son corps.



```python
test=[]
for i in range(500):
    col = []
    expected = test_outputs[i]
    predicted = machine_Translation_System(input_sequences_encoder[i:i+1])
    col.append(expected)
    col.append(predicted)
    if (expected.lower() == predicted.lower()):
        col.append(1)
    else:
        col.append(0)

    colwordnum = []
    colwordeq = []
    count=0
    temp_expect = expected.lower().split() 
    temp_predict = predicted.lower().split()

    if ( len(temp_expect) <= len(temp_predict)):
      lenw=len(temp_expect)
    else:
      lenw=len(temp_predict)
    

    for j in range (0,lenw):
      if (temp_predict[j]==temp_expect[j]):
        count+=1

    col.append(lenw)
    col.append(count)
    test.append(col)
print(test)
```

    [['Tom renomma le fichier.', 'tom sur le travail ?', 0, 4, 2], ["Tom dit qu'il n'a pas mang?? depuis trois jours.", "tom dit qu'il n'a pas mang?? depuis trois jours.", 1, 9, 9], ['Il peut parler cinq langues.', "il se faut que parler en monde d'un jours.", 0, 5, 1], ["Je suis pratiquement s??r que Tom n'a pas de fr??re.", "je suis s??r que tom n'a pas de fr??re.", 0, 9, 2], ["Je n'y pr??te pas la moindre attention.", 'je ne me sentais pas un peu !', 0, 7, 1], ["Ce n'est pas aussi simple qu'on le croit.", "ce n'est pas aussi facile ?? ??a.", 0, 7, 4], ["Je vous ai dit que c'??tait dangereux.", "je vous ai dit que c'??tait n'??tait tr??s fini.", 0, 7, 6], ["Elles ont essay?? de le d??courager d'y aller.", "ils ont essay?? de le lui d'y aller.", 0, 8, 6], ['??a tient debout.', '??a a vraiment du sens.', 0, 3, 1], ['Cet homme me dit quelque chose.', "??a a l'air quelque chose que dire !", 0, 6, 0], ["Quoi d'autre peut-il foirer ?", "?? quoi d'autre d'autre ?", 0, 5, 1], ["Tom en voulait un, mais il n'avait aucune id??e d'o?? en trouver.", "tom a dit qu'il avait aucune id??e ?? ce qu'il qu'il avoir ??t?? en train cela.", 0, 12, 1], ['Il est m??decin.', 'il est m??decin.', 1, 3, 3], ['Racontez-nous une histoire de fant??mes.', 'quel pouvait une monde est !', 0, 5, 0], ["Qu'est-ce qui t'a fait poser cette question ?", "qu'est-ce que tu as fait ??a ?", 0, 7, 1], ['Ne l??che pas le chien.', 'ne sois pas ?? la f??te !', 0, 5, 2], ['Vous devez travailler ensemble.', 'il faut que vous soyez plus parler.', 0, 4, 0], ['Je ne peux manger cela.', 'je ne peux manger cela.', 1, 5, 5], ['Voil?? le train.', 'o?? est la t??l??vision.', 0, 3, 0], ['Je ne veux pas ??tre un poids pour toi.', 'je ne veux pas ??tre un travail pour toi.', 0, 9, 8], ['Le train arrive au quai num??ro 5.', 'le train ?? train ?? passer o?? elle ?? vous maison un chambre jeune.', 0, 7, 2], ['Je ne parle pas de toi derri??re ton dos.', 'je ne suis pas de te dire ?? quoi ce que tu as !', 0, 9, 4], ['Tom a gar?? la voiture derri??re la maison de Mary.', 'tom a achet?? la voiture de la maison de la maison.', 0, 10, 7], ["Aujourd'hui, j'ai bon app??tit.", "j'ai beaucoup de bonnes fois pour vous.", 0, 4, 0], ['Nous jouons souvent aux ??checs.', 'nous jouons souvent aux ??checs.', 1, 5, 5], ["Je n'avais aucune intention de m'y rendre seul.", "je n'avais jamais pens?? de faire la question.", 0, 8, 3], ['Et alors ?', '??a a ?? quelque chose.', 0, 3, 0], ["Vous ??tes la plus jolie femme que j'ai jamais vue.", "vous ??tes la plus belle fille que j'ai jamais vue.", 0, 10, 8], ["Tom s'est d??barrass?? de sa vieille voiture.", 'tom a perdu de sa maison dans sa voiture.', 0, 7, 3], ['Vous ignorez ce que ceci signifie pour nous.', 'vous ne savez pas quoi quoi quoi vous parlez.', 0, 8, 1], ["J'ai des enfants.", "j'ai des enfants.", 1, 3, 3], ['Il y a des choses que vous devez savoir.', 'il y a des choses que vous devez faire ??a.', 0, 9, 8], ['Quand ??tes-vous arriv??es ?? Boston ?', 'quand ??tes-vous arriv??es ?? boston ?', 1, 6, 6], ["J'avais l'intention d'annuler votre rendez-vous d'aujourd'hui.", "je souhaite l'intention de votre jour sont demain.", 0, 6, 0], ["Je savais que c'??tait Tom.", "je savais que c'??tait tom.", 1, 5, 5], ['Je suis un de leurs amis.', 'je suis un de leurs amis.', 1, 6, 6], ["J'ai la dalle.", "j'ai tr??s faim.", 0, 3, 1], ['Il est rentr?? tard hier soir.', "il est rentr?? chez l'??cole en retard il soir.", 0, 6, 3], ["Je me sens seul lorsque tu n'es pas l??.", "je me sens si tu n'es pas l??.", 0, 8, 3], ['Elles ont appel?? ?? mettre fin au combat.', 'ils ont appel?? ?? mettre devant son pi??ce.', 0, 8, 4], ["J'ai connaissance de cette coutume.", 'je suis s??r de le vol.', 0, 5, 0], ['Entre la viande et le poisson, lequel pr??f??rez-vous\u202f?', 'que faites-vous du d??jeuner et elle.', 0, 6, 0], ["J'en ai fini de faire l'andouille.", "j'en ai fini de faire ??a.", 0, 6, 5], ["Rien ne pourrait l'arr??ter.", 'rien ne pourrait se passe.', 0, 4, 3], ['Nous ne sommes pas impressionn??es.', 'nous ne sommes pas fier', 0, 5, 4], ['Ne sois pas si pudique.', 'ne sois pas si en train !', 0, 5, 4], ['Cette voiture-ci est plus grande que celle-l??.', 'cette voiture est plus grande grande grande que celle-l??.', 0, 7, 4], ['Je fais les r??gles.', 'je les r??gles.', 0, 3, 1], ["J'ai besoin que tu viennes avec moi.", 'je dois vous me rendre chez moi.', 0, 7, 1], ["Il s'agissait d'un crime passionnel.", 'il ??tait un verre pour avoir du monde du ai avoir un verre ?? boire.', 0, 5, 1], ['Assurez-vous de sauvegarder tous vos fichiers.', 'assurez-vous de tous vos fichiers.', 0, 5, 2], ['Pourquoi ne remets-tu pas ta d??mission ?', 'pourquoi ne vous pas jamais.', 0, 5, 3], ["Je ne veux pas de f??te d'anniversaire cette ann??e.", "je ne veux pas un travail ce soit d'un ann??e.", 0, 9, 4], ['Tom a lev?? les yeux.', "tom a l'air de fr??re.", 0, 5, 2], ['??\xa0Aime-t-il la musique\xa0?\xa0?? ??\xa0Oui, il aime ??a.\xa0??', "cette lettre qu'il a pass?? ce qu'il qu'il a besoin ?", 0, 11, 0], ['Je suis mont?? dans le mauvais train.', 'je suis all?? ?? la premi??re vers le mauvais ce matin.', 0, 7, 2], ["On m'a dit de t'aider.", 'je suis dit pour vous aider.', 0, 5, 1], ["Je n'ai pas fait ??a hier donc je dois le faire aujourd'hui.", "je n'ai pas fait ??a que je dois le faire aujourd'hui.", 0, 11, 5], ['Je d??testerais voir cela se reproduire.', 'je ne vais te faire ??a cela a ceci.', 0, 6, 1], ["C'est mon CD.", "c'est mon patron.", 0, 3, 2], ['Veux-tu manger un morceau\u202f?', 'veux-tu manger quelque chose ?', 0, 5, 3], ['Ne trichez pas.', 'ne va pas.', 0, 3, 2], ['Je ne suis pas en col??re apr??s vous, seulement tr??s d????u.', 'je ne suis pas en col??re ?', 0, 7, 6], ['Elle nous a pr??par?? un en-cas.', 'elle nous a un parc.', 0, 5, 3], ['Il lui a serr?? la main.', 'il a son heures sur la main.', 0, 6, 1], ['Tom est plut??t intelligent.', 'tom est tr??s bien.', 0, 4, 2], ['Il pense vouloir devenir ing??nieur.', "il pense qu'il a un grand id??e.", 0, 5, 2], ['La rumeur ne reposait pas sur des faits.', 'la rumeur ne fut pas grand grand grand !', 0, 8, 4], ["Tom m'a dit qu'il se sentait seul.", "tom m'a dit qu'il se dit qu'il ??tait ??a.", 0, 7, 5], ['Tu as beaucoup de t??l??phones.', 'vous avez beaucoup de t??l??phones.', 0, 5, 3], ['Tu es un tr??s bon danseur.', 'vous ??tes un tr??s bon bon !', 0, 6, 3], ["L'avenir appartient ?? ceux qui se l??vent t??t.", 'le pont a pass?? ?? cause de retard.', 0, 8, 0], ["J'irai moi-m??me.", 'je me vais aller.', 0, 2, 0], ['Je ne me sens pas tr??s bien.', 'je ne suis pas tr??s en train de me dire comme ce que ??a.', 0, 7, 2], ["Il m'a fallu faire mon devoir.", "il m'a fallu faire mon travail.", 0, 6, 5], ["Je vous le demande en tant qu'amie.", "je vous le demande en tant qu'amie.", 1, 7, 7], ['Je ne me dispute pas.', 'je ne me souviens pas de son nom.', 0, 5, 3], ["J'aurais d?? faire ??a il y a longtemps.", "j'aurais d?? faire ??a il fait une id??e.", 0, 8, 5], ['Tu ne veux pas vraiment ??a, si ?', 'tu ne veux pas vraiment si ?', 0, 7, 5], ["Il n'arrive pas ?? expliquer ce qui est arriv??.", "il n'arrive pas ?? dire ce qui est arriv??.", 0, 9, 8], ["Vous ??tes en train de gagner, n'est-ce pas ?", 'vous ??tes en train de vous.', 0, 6, 5], ["Il fait de l'urticaire lorsqu'il mange des ??ufs.", "il fait de l'urticaire lorsqu'il mange des ??ufs.", 1, 8, 8], ['Je vais aller chez le dentiste demain.', 'je vais aller chez la t??l??vision.', 0, 6, 4], ['Vous savez de quoi je parle.', 'vous savez de quoi je parle.', 1, 6, 6], ['Quel vent !', 'quel vent !', 1, 3, 3], ['Il est malade.', 'il est malade.', 1, 3, 3], ["T'ont-ils bless??e ?", 'elles avez-vous eu ?', 0, 3, 0], ["J'aimerais prendre ??a avec moi.", "j'aimerais vous en faire avec moi.", 0, 5, 1], ['Qui a fait la photo ?', 'qui a fait la photo ?', 1, 6, 6], ['Est-ce que je te parais vieux\xa0?', 'est-ce que je te y a un chose ?', 0, 7, 4], ['Voulez-vous du caf??\u202f?', 'voudriez-vous du d??jeuner ?', 0, 4, 2], ["Je n'ai simplement pas voulu vous contrarier.", 'je ne voulais pas que tu sois simplement te tirer ici.', 0, 7, 2], ['Tout le monde ??tait impressionn?? par cette machine.', "tout le monde ??tait vraiment que j'ai ??t?? hier.", 0, 8, 4], ['Il se trouve que mon ordinateur ??tait en panne hier.', 'mon p??re ??tait en train de faire du mois dernier.', 0, 10, 0], ['Je p??le une orange pour toi.', 'je vais une grosse pour toi.', 0, 6, 4], ["J'appellerai si je trouve quoi que ce soit.", 'je vais quelque chose pour dire quelque chose.', 0, 8, 0], ['Je lui ai demand?? de me reconduire chez moi.', 'je lui ai demand?? de me rendre pour moi.', 0, 9, 7], ['On se voit une fois par mois.', 'nous pourrions le mois avant par un moins par cette ann??e.', 0, 7, 1], ['Tom est un type tr??s amical.', "tom est un type tr??s en chose de l'??cole.", 0, 6, 5], ['Cela ne te semble-t-il pas ??trange\xa0?', 'cela ne te te souviens pas y faire.', 0, 7, 3], ['Je me demande o?? il est maintenant.', 'je le demande o?? il est tout le monde.', 0, 7, 5], ["J'ai fait d??marrer le moteur.", "j'ai pris une t??te ?? fumer.", 0, 5, 1], ["Tom l'a reconnu.", "tom l'a reconnu.", 1, 3, 3], ["Qu'y a-t-il de mal ?? cela ?", "qu'est-ce qui se pourrait de ??a ?", 0, 7, 1], ['Tom a appris ?? skier tout seul.', 'tom a appris ?? tout tout seul.', 0, 7, 6], ["J'esp??re que vous avez trouv?? tout ce dont vous avez besoin.", "j'esp??re que vous avez trouv?? tout ce dont vous avez besoin.", 1, 11, 11], ['Ne sachant que r??pondre, je restai coi.', "ne crois que je n'ai pas fait mon probl??me avec temps.", 0, 7, 2], ['Que penses-tu du nouvel enseignant\u202f?', 'que pensez-vous de la premi??re ??tait de no??l ?', 0, 6, 1], ["J'ai pris plaisir ?? votre compagnie.", "j'ai pris votre r??gles.", 0, 4, 2], ["J'ai pass?? le test haut la main.", "j'ai pass?? le suis haut du haut de la pi??ce sont si dans votre si cher.", 0, 7, 4], ["J'ai vraiment besoin d'un verre.", "j'ai vraiment besoin d'un verre.", 1, 5, 5], ['Vous allez tous mourir.', 'vous allez tous mourir.', 1, 4, 4], ["C'est un courriel ind??sirable.", "c'est un verre ?", 0, 4, 2], ['As-tu vu Tom derni??rement ?', 'as-tu vu tom vu ?', 0, 5, 4], ["Si tu n'as pas cette cl??, tu ne pourras pas entrer.", 'si tu ne te pas pas, pas, si ??a ?', 0, 10, 2], ["Je veux seulement t'aider.", 'je veux ??tre pour toi.', 0, 4, 2], ["J'appr??cie vraiment ce gar??on.", 'je pensais que vous avez une chance.', 0, 4, 0], ['Je suis suffisamment vieux.', 'je suis suffisamment de petit p??re.', 0, 4, 3], ['Je viens de dire une chose tr??s stupide.', "je n'ai vraiment dit une tr??s stupide.", 0, 7, 2], ['Elle a march??, bras dessus bras dessous, avec son p??re.', "elle a laiss?? son bras avec son voiture et sa porte n'est pas en p??re.", 0, 10, 2], ["Est-ce que tu aimes les histoires d'aventure\xa0?", 'est-ce que tu aimes les histoires ?', 0, 7, 6], ['Cela ne me surprend pas.', '??a ne me me pla??t pas.', 0, 5, 2], ["Ma jauge d'essence est cass??e.", 'ma s??ur est cass??e.', 0, 4, 1], ["C'est ainsi qu'il en allait.", "c'est ainsi qu'il le sait.", 0, 5, 3], ["Tom s'est habitu?? ?? vivre ici.", "tom n'est pas all?? ?? vivre ici.", 0, 6, 1], ["J'ai entendu que l'on m'appelait.", "j'ai entendu que je suis entendu mon p??re.", 0, 5, 3], ["C'est ainsi que j'aime les choses.", "c'est ainsi que j'aime les choses.", 1, 6, 6], ["Je savais que ce n'??tait pas sain.", "je savais que ce n'??tait pas fini.", 0, 7, 6], ['Est-ce que tu es une bonne nageuse\xa0?', 'est-ce que tu es une bonne bonne ?', 0, 8, 7], ['Le pique-nique a ??t?? annul?? ?? cause de la pluie.', "le train a ??t?? par ?? cause de ce qu'il n'a pas sur hier.", 0, 10, 6], ['Voudriez-vous bien arr??ter de me demander cela ?', 'voudriez-vous bien arr??ter de me dire cela ?', 0, 8, 7], ['Je ne me souviens de rien de plus maintenant.', "je ne me souviens pas de dire que je n'ai rien ?? ??a.", 0, 9, 4], ['Ils vivent dans la crainte constante des inondations.', 'ils vivent dans la rivi??re.', 0, 5, 4], ["Je ne sais pas d'o?? ils viennent.", 'je ne sais pas ils aux r??gles.', 0, 7, 4], ["C'est hilarant.", "c'est ??a ?? nouveau.", 0, 2, 1], ["Tom n'a que trois semaines.", 'tom est trois fois de marie.', 0, 5, 1], ['Il exigea une augmentation.', 'il prit ??chou??.', 0, 3, 1], ["N'oubliez jamais cela.", 'ne jamais jamais cela.', 0, 3, 1], ['Savez-vous faire du v??lo ?', 'pouvez-vous ??tre un v??lo ?', 0, 5, 2], ['Il se place toujours en premier.', 'il a toujours le match de nom.', 0, 6, 1], ['Es-tu s??r que tu veux quitter ton emploi ?', 'es-tu s??r que tu veux quitter ton emploi ?', 1, 9, 9], ["Combien d'ann??es Tom a-t-il pass?? ?? Boston ?", "combien d'ann??es tom a-t-il pass?? ?? boston ?", 1, 8, 8], ['Vous devez conna??tre Tom assez bien.', 'vous devriez tom dit que tom bien.', 0, 6, 1], ['Portes-tu des lunettes ?', 'avez-vous des id??es ?', 0, 4, 2], ["Je pensais que tu avais besoin d'argent.", "je pensais que tu avais besoin d'argent.", 1, 7, 7], ["C'est un nouveau livre.", "c'est un nouveau livre.", 1, 4, 4], ['Je ne tiens pas en place.', 'je suis en train de feu.', 0, 6, 1], ['Les hommes y sont meilleurs que les femmes.', 'les hommes sont que les meilleurs plus amis.', 0, 8, 2], ['Tu r??ussiras si tu essayes.', 'tu ferais si tu te donnerai tu le montre.', 0, 5, 3], ['Qui a dit ?? Tom de partir\u202f?', 'qui a dit ?? tom de mary.', 0, 7, 6], ["La raison pour laquelle Tom ne peut pas y aller est qu'il n'a pas d'argent.", 'la raison pour la raison pour tom ne pas ??tre trop pour toi.', 0, 13, 3], ['Combien tout ceci a-t-il co??t??\u2009?', 'combien tout ??a a-t-il eu une bonne ?', 0, 6, 3], ['Fais ce que tu veux.', 'fais ce que tu veux.', 1, 5, 5], ['Y a-t-il la version sous-titr??e ?', 'y a-t-il la version des pont ?', 0, 6, 4], ["C'est pourquoi Tom et moi ??tions d????us.", "c'est pourquoi tom et moi ??tions occup??.", 0, 7, 6], ['Quel pays !', 'quel pays !', 1, 3, 3], ['Il nous fallut arr??ter.', 'il nous faut arr??ter.', 0, 4, 3], ['??tes-vous ?? la maison ?', '??tes-vous ?? la maison ?', 1, 5, 5], ["Nous aurons tous faim alors assure-toi d'apporter suffisamment de nourriture pour tout le monde.", "nous avons tous faim tout ne t'ai fait pour tout ce que tu es pour toi.", 0, 14, 3], ["Je vais t'accompagner jusqu'?? l'intersection.", "je vais ??tre rendrai jusqu'?? un jour ?", 0, 5, 2], ['Te demandes-tu ce que nous avons fini par faire ?', '??tes-vous s??r que nous ne nous sommes plus ?', 0, 9, 0], ['??tes-vous occup??es vendredi soir ?', '??tes-vous encore ici du nuit ?', 0, 5, 1], ['Tu entres dans la trentaine.', 'tu es la journ??e.', 0, 4, 1], ['Nous sommes rest??es ensemble par mesure de s??curit??.', 'nous avons ??t?? par par temps.', 0, 6, 2], ['R??trospectivement, on peut voir les choses qui auraient d?? ??tre faites diff??remment ou pas du tout.', 'on peut les choses que tu as d?? faire des choses ne pas ?? faire des photo tu es du faire.', 0, 16, 0], ["Merci beaucoup d'avance pour ta coop??ration.", 'merci de venir ta as !', 0, 6, 1], ["Pour le moment, j'??tudie le fran??ais dans cette ??cole de langues.", 'pour le moment, je suis ?? la ville dans un nouveau fois ?? un p??re.', 0, 11, 3], ['Elle a cuisin?? une tarte aux pommes pour son mari.', 'elle a son pommes du son pommes ?? pommes pour son mari.', 0, 10, 3], ['Les temps sont durs.', 'ce sont plus de temps que les semaine ?', 0, 4, 0], ['Est-ce tout faux ?', 'est-ce tout cela ?', 0, 4, 3], ["Tom gagne beaucoup d'argent.", "tom beaucoup beaucoup d'argent.", 0, 4, 3], ['Vous ??tes vraiment cingl??s.', 'tu es vraiment du ?', 0, 4, 1], ['Il tint sa promesse.', 'il il a perdu son lit.', 0, 4, 1], ["En plus de l'anglais, elle parle couramment fran??ais.", 'en plus de parle elle parle en fran??ais.', 0, 8, 6], ['Nous nous en approchons.', 'nous nous en approchons.', 1, 4, 4], ['Qui a organis?? ce voyage\u202f?', 'qui a dit ce qui se passe.', 0, 6, 3], ["C'est ce que je dis depuis le d??but.", "c'est ce que je suis depuis le train de la journ??e.", 0, 8, 6], ['Les portes sont ferm??es.', 'les deux sont sont ferm??es.', 0, 4, 2], ['Est-ce que tu as quelque chose pour le mal de t??te\xa0?', 'as-tu une chose ?? propos de ce qui ?', 0, 9, 0], ["J'ai pens?? que je pourrais compter sur vous.", "j'ai pens?? que je pourrais avoir besoin de vous.", 0, 8, 5], ["J'esp??re que nous aurons un No??l blanc.", "j'esp??re que nous avons un peu plus bon blanc.", 0, 7, 4], ["Il est plus intelligent qu'eux.", 'il est plus intelligent que les animaux intelligent de que les autres.', 0, 5, 4], ['Selon les journaux, il neigera demain.', "comment sera la semaine n'est-ce ?? nouveau.", 0, 6, 0], ['??a lui a fait une peur bleue.', '??a le monde a peur ?? la peur', 0, 7, 1], ["Suis-je oblig?? d'??tre hospitalis??\u202f?", "suis-je oblig?? d'??tre une train ?", 0, 5, 3], ["C'est une ??tudiante qui ??tudie s??rieusement.", "c'est une fille qui est plus en p??re.", 0, 6, 3], ["Je commence ?? m'habituer ?? la nourriture d'ici.", 'je vais passer ?? seul pour la semaine ici.', 0, 8, 1], ['Ce serait vraiment int??ressant.', 'ce serait vraiment int??ressant.', 1, 4, 4], ['Le Nozomi est le plus rapide de tous les trains au Japon.', 'le train est de le plus qui est tous les gens que les monde.', 0, 12, 2], ["Elles disposent de plein d'argent.", "elles disposent de plein d'argent.", 1, 5, 5], ["C'est comme ??a.", "c'est la seule ?? la plus sujet.", 0, 3, 1], ['Sa montre a dix minutes de retard.', 'sa montre a dix minutes de retard.', 1, 7, 7], ['Es-tu v??g??tarienne ?', '??tes-vous es-tu ?', 0, 3, 1], ["Vous m'avez beaucoup appris.", "vous m'avez beaucoup quelque chose.", 0, 4, 3], ["Qu'ai-je loup?? ?", "qu'est-ce que vous avez fait ?", 0, 3, 0], ['Tous les fauteuils sont occup??s.', "tous les deux sont d'entre les deux sont de nourriture.", 0, 5, 3], ["Qu'arrive-t-il ensuite ?", "qu'est-ce que tom va bien\xa0?", 0, 3, 0], ["C'est quoi ce bidule ?", "c'est ce qu'il est ?? la sant??.", 0, 5, 1], ['Je peux te d??pecer ?? mains nues.', 'je peux te peux tes mains sont mes mains sont mes yeux pour vos enfants.', 0, 7, 4], ['??a para??t amusant ?? faire.', 'on peut faire la raison ?? faire.', 0, 5, 0], ['Voulez-vous passer un march?? ?', 'veux-tu prendre un verre ?', 0, 5, 2], ['Je suis trop occup??e.', 'je suis trop occup??e.', 1, 4, 4], ['Cette jeune personne est infirmi??re.', 'cette jeune personne est infirmi??re.', 1, 5, 5], ["C'est plus que je n'escomptais.", "c'est plus que je n'en plus dire.", 0, 5, 4], ['Pourquoi quiconque voudrait nager dans cette rivi??re ?', 'pourquoi quiconque voudrait nager dans cette rivi??re ?', 1, 8, 8], ["N'avancez pas.", 'restez tranquille !', 0, 2, 0], ["On n'y peut rien.", '??a ne peut pas ??tre ?? nourriture.', 0, 4, 1], ["Qui t'a envoy?? ?", 'qui vous a dit ?', 0, 4, 1], ['Le temps de me rendre compte, je ne pouvais plus voir les oiseaux.', 'le fois que je ne voulais pas me faire la plus de temps.', 0, 13, 1], ['Nous avons trouv?? quelque chose.', 'nous avons trouv?? quelque chose.', 1, 5, 5], ['Tu devrais vraiment participer au concours.', 'vous devez vraiment le chien.', 0, 5, 1], ['Vous devez passer une audition avant de pouvoir rejoindre le ch??ur.', 'vous devez passer un endroit de toi en rendre avant ?', 0, 11, 3], ["Je n'ai pas ??t?? dipl??m??.", "je n'ai pas ??t?? impressionn??.", 0, 5, 4], ["C'est le livre que j'ai achet?? hier.", "c'est le livre que j'ai achet?? du no??l.", 0, 7, 6], ['Je ne vous ai m??me pas remarqu??.', 'je ne vous ai m??me pas dit que moi.', 0, 7, 6], ['Tom a devin??.', 'tom a reconnu.', 0, 3, 2], ['Elles ont constamment continu?? ?? parler.', 'elles ont continu?? ?? parler.', 0, 5, 2], ['Comment puis-je quitter ce travail\u202f?', 'comment puis-je vous voulez cette photo ?', 0, 6, 2], ["On m'a tendu un pi??ge.", "on m'a une ??tait sur de sa musique.", 0, 5, 2], ['Cette lettre vous est adress??e.', 'cette lettre vous est en fois ?? ton aide.', 0, 5, 4], ['Il a perdu de vue cet oiseau.', 'il a perdu de cet r??ve cet oiseau.', 0, 7, 4], ["Il est devenu p??le lorsqu'il a entendu ces nouvelles.", "il s'est pass?? du lorsqu'il a entendu ce qu'il s'est fatigu??.", 0, 9, 4], ['Je suis d??sol?? de ne pas avoir pu assister ?? votre f??te.', 'je suis s??r que je ne suis pas votre f??te pour votre aide.', 0, 12, 2], ["M??re se rend ?? l'h??pital dans la matin??e.", "m??re se rend ?? l'??cole ?? la f??te de un bus.", 0, 8, 5], ['Regardons la situation dans son ensemble.', 'regardons la pi??ce dans un de son chance.', 0, 6, 3], ['Je veux regarder ceci.', 'je veux que ??a se fais.', 0, 4, 2], ['Nous sommes confront??s ?? un tas de probl??mes.', 'nous sommes un de temps ?? notre maison ?? lumi??re.', 0, 8, 2], ['Il a tr??s peur des chiens.', 'il a tr??s peur des chiens.', 1, 6, 6], ['Noue ton lacet de chaussure.', 'va ton arme !', 0, 4, 1], ["Peut-on s'en aller ?", 'pouvons-nous y aller ?', 0, 4, 2], ["Aimez-vous qu'on vous fasse attendre ?", 'veux-tu ??tre rendre ?', 0, 4, 0], ['As-tu une assurance m??dicale ?', 'avez-vous une nouvelle ?', 0, 4, 1], ['Soigne-toi. Ne tombe pas malade.', 'ne tombe pas malade.', 0, 4, 0], ['Elles te craignaient.', 'elles te pouvez ?', 0, 3, 2], ['Mon nouvel emploi me laisse peu de temps pour rencontrer des gens.', 'mon p??re me dois peu de temps pour moi.', 0, 9, 1], ['Je vous rembourserai d??s que je le peux.', "je vous prie de l'argent plus que je peux.", 0, 8, 2], ['En Grande-Bretagne, les tartes ?? la viande hach??e sont traditionnellement consomm??es au moment de No??l.', 'en certains les animaux ont d?? la nuit sont pas bon pour ce semaine.', 0, 14, 3], ['Ce sont des d??tritus.', "c'est du toi.", 0, 3, 0], ["Il s'est bless?? le genou lorsqu'il est tomb??.", "il s'est bless?? le lorsqu'il est imm??diatement.", 0, 7, 4], ['Est-ce que vous avez un avocat\xa0?', 'est-ce que tu as un coup de coup ?', 0, 7, 3], ["Je te le demande en tant qu'ami.", "je vous le demande en tant qu'amie.", 0, 7, 5], ["Juste ?? ce moment-l??, j'ai entendu des pas dans la cage d'escalier.", "?? ce que je l'ai entendu des pi??ce dans la pi??ce de la pi??ce se maison ?", 0, 12, 4], ['Tom entre dans sa p??riode de pubert??.', 'tom entre dans sa p??riode de feu.', 0, 7, 6], ['Nous avons encore un long chemin ?? parcourir.', 'nous avons encore un temps ?? faire pour moi.', 0, 8, 4], ["Ceci est une voiture import??e d'Allemagne.", "c'est une voiture de voiture est en voiture ?? voiture de voiture ?", 0, 6, 0], ['Tu as ??t?? mon ami.', 'vous avez ??t?? mon ami.', 0, 5, 3], ["Il faut que j'y aille.", 'il me faut aller aller !', 0, 5, 1], ['Mon c??ur souffre pour ces enfants qui meurent de faim.', 'mon c??ur souffre pour les enfants qui pas les enfants s??r.', 0, 10, 6], ['Comment ??tes-vous arriv??es ici avant moi ?', 'comment as-tu pu dire avant cette nuit ?', 0, 7, 2], ['Mon permis de conduire expire la semaine prochaine.', 'mon permis de conduire la semaine prochaine.', 0, 7, 4], ["N'est-ce pas horrible\xa0?", '??a ne suis pas ?? grand train de que vous plait.', 0, 4, 0], ['Il me faut davantage de lumi??re.', "j'ai besoin de nombreux carte !", 0, 6, 0], ['On doit ??tre responsable de sa propre conduite.', 'on doit ??tre responsable de son propre conduite.', 0, 8, 7], ["C'est la maison de mon p??re.", "c'est ma maison de mon p??re.", 0, 6, 5], ['Ne retire pas encore la prise !', 'ne laissez pas la personne !', 0, 6, 2], ["Vint l'automne et les feuilles se mirent ?? tomber.", 'les nouvelle et se vacances ?? cause ?? la semaine pour ce matin.', 0, 9, 2], ["??coute, c'est mon probl??me !", "c'est mon probl??me au temps.", 0, 5, 0], ['Tom ??tait-il s??duisant ?', 'tom ??tait tom.', 0, 3, 1], ['Quel est le pire tatouage que vous ayez jamais vu\u202f?', 'quel est le pire chose que vous ayez jamais ??t?? jamais ?', 0, 11, 8], ['Personne ne nous ??coutait.', 'personne ne nous parler.', 0, 4, 3], ["Je n'ai aucune chance.", "je n'ai pas besoin de chance.", 0, 4, 2], ["C'est, en v??rit??, assez embarrassant.", "c'est du probl??me ?? assez sens.", 0, 5, 0], ['Tout le monde a le droit ?? sa propre opinion. Cependant, il est parfois pr??f??rable de ne partager cette opinion avec personne.', 'tout le monde a le probl??me ?? un propre pour personne ne fait pas ??tre pr??f??rable de ne peux pas sortir.', 0, 21, 7], ["Combien de personnes y avait-il dans l'avion ?", 'combien de personnes ont eu le australie ?', 0, 8, 4], ["Elle a couru aussi vite qu'elle a pu.", "elle a aussi vite qu'il est aussi vite", 0, 8, 2], ["Tout ce que je sais, c'est qu'il vient de Chine.", 'tout ce que je ne suis plus ?? boston de temps.', 0, 10, 4], ['Il joue au tennis trois fois par semaine.', 'il il y a trois trois fois par semaine.', 0, 8, 2], ["J'ai d??couvert ton sale petit secret.", "j'ai toujours votre as du secret.", 0, 6, 2], ['Je bois mon th?? sans sucre.', "je bois mon bois pour j'ai du matin.", 0, 6, 3], ['Il ne sait pas distinguer le bien du mal.', 'il ne sait pas que je le lui a pas.', 0, 9, 4], ['Je suis en quatri??me.', 'je suis en suis en train de boston.', 0, 4, 3], ['Tom ne mourra pas de faim.', 'tom ne se pas pas de rester.', 0, 6, 3], ['Tom veut vraiment ??tre ton ami.', 'tom veut vraiment ??tre ton ami.', 1, 6, 6], ["Cette ville n'a pas beaucoup chang?? au cours des dix derni??res ann??es.", "cette ville n'a pas beaucoup au beaucoup de temps.", 0, 9, 5], ["Je ne t'ai pas demand?? de venir ici.", 'je ne me souviens pas pas venir de venir ici.', 0, 8, 3], ["Tom s'est lev?? et a march?? vers la porte.", 'tom a essay?? de se promener et nous a march?? au de la porte.', 0, 9, 1], ['Quand est-ce survenu ?', 'quand est-ce que ??a a eu ?', 0, 4, 2], ["Pourquoi n'??tes-vous pas all?? voir la police\xa0?", 'pourquoi ne voulez pas all?? la nuit en train de main le maison ?', 0, 8, 2], ['Elle loua un appartement de quatre pi??ces.', 'elle a un appartement de quatre dans le enfants dans le enfants.', 0, 7, 5], ['Tu es le fils de qui ?', 'qui est toujours ?? vous ?', 0, 6, 0], ['De tous mes amis, il est le plus proche.', 'tous mes mes il sont d??j?? chez de la nuit.', 0, 9, 1], ["J'admets avoir fait ??a.", 'je pensais que ??a le faire ??a.', 0, 4, 0], ['As-tu vu mes cl??s ?', 'as-tu vu mes cl??s ?', 1, 5, 5], ['Nous avons tout ce dont nous avons besoin maintenant.', 'nous avons tout ce dont nous avons besoin maintenant.', 1, 9, 9], ['Ce fut un super voyage.', "c'??tait un bon voyage.", 0, 4, 0], ["J'ai fait tout ce que j'ai pu pour sauver Tom.", "j'ai fait tout ce que j'ai pu pour sauver tom.", 1, 10, 10], ['Je dois trouver ma clef.', 'je dois me devrais les pi??ce au chien.', 0, 5, 2], ['Fais-moi voir ??a.', 'laisse-moi voir ??a.', 0, 3, 2], ["Je crois que c'est la mienne.", "je crois que c'est la v??rit??.", 0, 6, 5], ['Elle habite ?? New York.', 'elle vit ?? new york.', 0, 5, 4], ['Vous y allez souvent\xa0?', 'est-ce que tu y aller ?', 0, 5, 0], ['Tom ne sait pas encore tr??s bien nager.', 'tom ne sait pas encore tr??s en nager.', 0, 8, 7], ['Tom sentit son t??l??phone vibrer.', 'tom t??l??phone son t??l??phone ?', 0, 5, 3], ["Je n'ai pas sauvegard?? le document.", "je n'ai pas sauvegard?? le probl??me de la r??union.", 0, 6, 5], ['Un miroir refl??te la lumi??re.', 'un prix est des lumi??re.', 0, 5, 2], ["Tom s'est pr??sent??.", 'tom a d??c??d??.', 0, 3, 1], ['Il tua cet homme.', 'il a eu un p??re.', 0, 4, 1], ["Il y a autre chose dont j'ai besoin de parler avec toi.", "il y a quelque chose dont j'ai besoin de toi.", 0, 10, 8], ['Elle a sugg??r?? que je lui ??crive imm??diatement.', 'elle a demand?? de lui ne se aller en lit.', 0, 8, 2], ["On m'a mis une prune.", "on m'a mis une mis en train de lit.", 0, 5, 4], ['Ils rel??ch??rent le prisonnier.', "ils l'ont sur le monde.", 0, 4, 1], ["Je pense qu'il est temps que je discute du probl??me avec elle.", "je pense qu'il est temps que je pourrais passer du temps.", 0, 11, 7], ["Vous avez bu une bi??re pendant le d??jeuner, n'est-ce pas\xa0?", "vous avez bu une journ??e pendant le d??jeuner, n'est-ce pas\xa0?", 0, 11, 10], ['Il soutenait que cette immense propri??t?? ??tait ?? sa disposition.', "il qu'il ??tait ?? cette biblioth??que.", 0, 6, 1], ["Je ne savais pas que le tofu c'??tait aussi bon que ??a.", 'je ne pense pas que ce soit la bonne id??e.', 0, 10, 4], ['Elle est blonde.', 'elle est un homme est en train de lit.', 0, 3, 2], ["C'est notre premier No??l ici en Australie.", "c'est notre premier ici est en train en train de main difficile.", 0, 7, 4], ['Il est influent.', 'il est influent.', 1, 3, 3], ['Ce mince livre est le mien.', 'ce livre est ?? la mien.', 0, 6, 2], ['Je ne sais pas ce que tu fais.', 'je ne sais pas ce que tu fais.', 1, 8, 8], ["Je n'avais jamais mang?? de nourriture chinoise auparavant.", 'je ne suis jamais mang?? de nourriture dans le lait.', 0, 8, 1], ["C'est un sujet compliqu??.", "c'est un sujet compliqu??.", 1, 4, 4], ['Quand bien m??me Tom ??tait mon meilleur ami, je commence ?? le ha??r.', 'si tom est mon derni??re fois pour le temps pour la journ??e de pas la nuit ?', 0, 13, 0], ["J'attends avec impatience de les voir ce printemps.", 'je me r??jouis de se revoir.', 0, 6, 1], ['Peu importe ce que tu dis, je ne pense pas que Tom est un gars sympa.', 'si vous dit que je ne pense pas que ce soit quelque chose est en chose est arriv??.', 0, 16, 1], ['Il me regarda pour une explication.', 'il me regarda pour une id??e.', 0, 6, 5], ['Si tu ??tais un espion essayant de te faire passer pour un natif et que tu le disais de cette mani??re, il est probable que tu te ferais prendre.', 'si tu ??tais un espion essayant de te passer un seul pour ce qui tu suis pas du week-end dernier.', 0, 20, 8], ['Je vous ai cru.', 'je vous ai ??a.', 0, 4, 3], ["La derni??re fois que je suis all?? ?? la plage, j'ai ??t?? gravement br??l?? par le soleil.", 'la fois que je suis all?? ?? la premi??re fois que je suis all?? en fois que je suis jamais la journ??e.', 0, 17, 1], ["J'essayai de le remonter.", "j'ai essay?? de le remonter.", 0, 4, 0], ['Je ne veux pas que Tom la voie.', 'je ne veux pas que tom a dit ??a.', 0, 8, 6], ['Il se met rarement en col??re.', 'il fait toujours sa nuit.', 0, 5, 1], ["Tom dit ?? Mary qu'il avait ?? lui parler.", "tom dit qu'il ?? mary ?? lui parler.", 0, 8, 2], ["J'ai pens?? que ??a pourrait ??tre vous.", 'je pensais que tu pourrais y aller.', 0, 7, 1], ["Qu'est-ce que tu pr??f??res dans la p??riode de No??l ?", "qu'est-ce que tu ferais dans la p??riode de no??l ?", 0, 10, 9], ["J'ai trop dormi parce que je me suis couch?? tard.", "j'ai simplement parce que je ne me suis pas en train de mon p??re.", 0, 10, 3], ['Conduis-toi selon ton ??ge.', 'comment est ton ??ge.', 0, 4, 2], ["Est-ce que tu t'es lav?? les mains\u202f?", 'avez-vous perdu votre chambre ?', 0, 5, 0], ['Pourquoi ??tes-vous si n??gatif ?', 'pourquoi ??tes-vous si en train si ?', 0, 5, 3], ['Comment avez-vous perdu votre bras ?', 'comment avez-vous perdu votre bras ?', 1, 6, 6], ['Comment cela se pourrait-il ?', 'comment cela cela ?', 0, 4, 2], ["Je ne dispose pas de l'argent pour acheter ce livre.", "je ne dispose pas de l'argent ?? acheter ce livre.", 0, 10, 9], ["Je n'arrive pas ?? croire que je vous embrasse.", "je n'arrive pas ?? croire que je vous en ??tes ??a.", 0, 9, 8], ['Je suis une patiente.', 'je suis une de amie.', 0, 4, 3], ['Arrivez-vous ?? voir cette petite maison ?', 'pouvez-vous voir en en temps.', 0, 5, 0], ['Le roi a enlev?? ses v??tements.', 'le chat a enlev?? ses v??tements.', 0, 6, 5], ["??a a l'air urgent.", "??a a l'air urgent.", 1, 4, 4], ['Ils ont ??t?? victorieux.', 'ils ont ??t?? victorieux.', 1, 4, 4], ['Pour combien de temps environ le voulez-vous\u202f?', 'combien de temps as-tu cong?? ?', 0, 6, 0], ['Est-ce que vous avez d?? attendre longtemps\xa0?', 'est-ce que vous avez pass?? un temps pour me suis t??t.', 0, 8, 4], ['Je serai de retour ?? six heures.', 'je serai ?? six heures de six heures.', 0, 7, 2], ["C'est la derni??re mode.", "c'est la derni??re chose est ?? pass??.", 0, 4, 3], ['Avez-vous bien dormi ?', 'as-tu bien dormi ?', 0, 4, 3], ["Je n'ai pas enfreint la loi.", "je n'ai pas pris la r??union.", 0, 6, 4], ['Les prix continuaient de monter.', 'les prix de dollars.', 0, 4, 2], ['Voudriez-vous tous vous d??tendre ?', 'voudriez-vous tous vous d??tendre ?', 1, 5, 5], ["Je pensais que c'??tait votre boulot.", "je pensais que c'??tait votre boulot.", 1, 6, 6], ["Restez tranquille s'il vous plait.", "restez s'il te plait.", 0, 4, 1], ["Pourquoi ne m'avez-vous pas appel?? la nuit derni??re ?", "pourquoi ne m'avez-vous pas appel?? la nuit derni??re ?", 1, 9, 9], ["J'ai commis les m??mes erreurs que la derni??re fois.", "j'ai eu la premi??re plus que ce que j'ai d?? la plus fois.", 0, 9, 2], ['Laisse-moi te donner un conseil.', 'laisse-moi te donner un conseil.', 1, 5, 5], ["Je n'avais pas id??e que tu savais jouer au mah-jong.", "je n'avais pas pens?? que tu savais ?? la question.", 0, 10, 6], ['Je reconnais l?? son ??uvre.', 'je peux voir son p??re.', 0, 5, 2], ["J'ai pass?? un bon moment ici.", "j'ai pass?? un bon ici.", 0, 5, 4], ["Avez-vous de l'aspirine avec vous ?", 'avez-vous de vous sur le journal ?', 0, 6, 2], ['Il est mort rapidement apr??s son accident.', "il est mort pendant qu'il se accident.", 0, 7, 4], ['Je suis s??re de pouvoir le faire.', 'je suis s??r que ??a soit si ??a.', 0, 7, 2], ['Tout le monde ??tait heureux.', 'tout le monde ??tait heureux.', 1, 5, 5], ["J'ai besoin de sucre.", "j'ai besoin de me besoin de la maison", 0, 4, 3], ['Essaye de r??sister.', 'essaye de r??sister.', 1, 3, 3], ['O?? sont les chaussures\u202f?', 'o?? sont les chaussures\u202f?', 1, 5, 5], ["Je n'ai pas beaucoup dormi cette nuit, du coup j'ai somnol?? toute la journ??e.", "je n'ai pas beaucoup dormi cette lettre en n'ai toute la journ??e.", 0, 12, 6], ['Tom a 13 ans, mais il croit toujours au P??re No??l.', "tom est en ans que il se croit que j'ai achet?? ce qu'il a dit.", 0, 11, 2], ['Il a finalement atteint ses objectifs.', 'il a finalement atteint ses objectifs.', 1, 6, 6], ['Les taxis sont chers.', 'les sont sont sont rendrai pla??t.', 0, 4, 2], ['Nous ne resterons plus dans cet h??tel.', "nous n'avons pas pu aller en train ?? boston du vie.", 0, 7, 1], ['La plupart des ??crivains sont sensibles ?? la critique.', 'la plupart des gens sont sensibles ?? que la plupart de vie.', 0, 9, 6], ['La nourriture est toujours insuffisante dans cette r??gion.', 'la vie est toujours dans cette r??gion.', 0, 7, 3], ["J'esp??re que le temps sera bon.", "j'esp??re que la vie sera bon.", 0, 6, 4], ["Quel est l'int??r??t de venir ici\xa0?", 'quel est ici de venir ici ?', 0, 7, 6], ['Puis-je emprunter votre stylo\xa0?', 'puis-je emprunter votre voiture\u202f?', 0, 5, 4], ['Tom est cens?? faire ??a sans aucune aide.', 'tom est en raison ?? ??a ??a se devriez cela.', 0, 8, 2], ["Elle l'??coute alors que personne d'autre ne le fait.", 'elle dis plus que je ne lui en fait plus de lui.', 0, 9, 2], ['Les quakers pensent que tous les hommes sont ??gaux.', 'les plupart des gens que nous ont tous tous livre.', 0, 9, 1], ['Les r??ves deviennent r??alit??.', 'les r??ves se vais mal.', 0, 4, 2], ["J'aime ce que vous avez fait avec vos cheveux.", "j'aime ce que tu sais avec faire ce n'??tait pas partir.", 0, 9, 3], ['Je ne suis pas du tout surpris.', 'je ne suis pas de me le peu la pi??ce de lumi??re.', 0, 7, 4], ['Dites-moi comment vous vous appelez.', 'dites-moi comment vous avez dit de votre v??rit??.', 0, 5, 3], ["Tom a l'air d'avoir faim.", "tom a l'air faim.", 0, 4, 3], ['Comment lisez-vous cet id??ogramme ?', 'comment rendez-vous cette lettre ?', 0, 5, 2], ["Elle n'appr??ciait pas de vivre en ville.", 'elle ne pas vivre en ville.', 0, 6, 2], ['Aviez-vous un livre favori quand vous ??tiez enfant ?', 'as-tu un livre que je pourrais ??tre un p??re ?', 0, 9, 2], ['Je me sens mal.', 'je me sens juste !', 0, 4, 3], ['Le chiot la l??cha ?? la joue.', 'le chat a son bouton ?? la main.', 0, 7, 1], ["Tom a viss?? l'ampoule.", "tom a l'ampoule.", 0, 3, 2], ["Je veux m'assurer que tu es celui que tu dis ??tre.", "je veux m'assurer que vous ??tes celui que vous seriez ??a.", 0, 11, 6], ["Son fr??re va ?? l'??cole en bus.", "son p??re est ?? l'??cole ?? l'??cole.", 0, 7, 3], ['Je suis d??sesp??r??e.', 'je suis que tu es du temps.', 0, 3, 2], ['R??pondez ?? ma question !', 'laisse ma question !', 0, 4, 0], ["Tom m'a agress??.", "tom m'a m'a ?", 0, 3, 2], ['Je vous conseillerais fortement de faire ce que le patron vous a dit de faire.', "je vous dois ??a ?? faire ce que vous fait ce n'est pour faire.", 0, 14, 5], ["Ils l'ont d??masqu??.", "ils l'ont l'ont", 0, 3, 2], ['Il a bu une bouteille de lait enti??re.', 'il a fait une bouteille de lait dans quelque chose.', 0, 8, 6], ['Je ne peux vivre sans elle.', 'je ne peux vivre sans elle.', 1, 6, 6], ["Je n'ai jamais pens?? que je te verrais ?? nouveau un jour.", "je n'ai jamais pens?? que je te crois ?? cette nouveau est au jour.", 0, 12, 8], ['C?????tait qui, la femme avec qui t?????tais hier ?', 'qui vous dit la femme avec la femme', 0, 8, 0], ['Des examens plus pouss??s sont requis.', 'des examens plus sont pouss??s', 0, 5, 3], ["Si j'??tais toi, j'y penserais ?? deux fois avant d'emprunter cette voie.", "si j'??tais toi, je veux cette derni??re fois ?", 0, 9, 4], ['Il avait une d??sagr??able voix per??ante.', 'il avait une une chance.', 0, 5, 3], ['Ne quittez pas !', 'ne va pas !', 0, 4, 3], ["Je n'aurais pas pu r??ver d'un meilleur cadeau pour ce No??l.", "je n'aurais pas pu ??tre aussi pour un cadeau pour ce no??l.", 0, 11, 4], ['Pourquoi me fixe-t-il ?', 'pourquoi me fixe-t-il ?', 1, 4, 4], ['Il manque trois boutons sur mon manteau.', 'il manque trois manque sur mon manteau.', 0, 7, 6], ['As-tu entendu ??a ?', 'as-tu entendu ??a ?', 1, 4, 4], ['Tom dira probablement non.', 'tom pourrait que ??a pourrait te mal.', 0, 4, 1], ['Tom ne portait pas de chaussures.', "tom n'a pas de chaussures.", 0, 5, 1], ["J'allais p??cher sous la glace, lorsque j'??tais plus jeune.", 'je suis habitu?? ?? vivre pendant pendant que je pourrais une voiture.', 0, 9, 0], ["Tu n'as nulle part o?? te cacher.", "vous n'avez nulle part o?? vous cacher.", 0, 7, 4], ['Il a mon ??ge.', 'il est mon ??ge.', 0, 4, 3], ['Combien de fr??res as-tu\u202f?', 'combien de temps avez-vous ?', 0, 5, 3], ['Je peux courir.', 'je vais prendre la d??faite.', 0, 3, 1], ['Nous avons assez de provisions pour tenir trois mois.', 'nous avons eu de temps ?? trois mois.', 0, 8, 3], ["L'aide internationale du Japon diminue en partie ?? cause d'un ralentissement de l'??conomie int??rieure.", 'en joue en retard de suis du japon du japon est en retard ?? ce que tu te est chez moi.', 0, 14, 0], ["Tom a enfreint les r??gles et a ??t?? exclu de l'??quipe.", 'tom a essay?? et les police et a ??t?? la fen??tre.', 0, 11, 2], ['Tu es beau.', 'tu es aussi beau.', 0, 3, 2], ['Les enfants dont les parents sont riches ne savent pas g??rer leur argent.', 'les enfants qui sont essay?? de ne pas pouvais peuvent faire si tu veuillez devriez pas.', 0, 13, 2], ['O?? avez-vous appris cela ?', 'o?? as-tu eu ce cela ?', 0, 5, 1], ['Le peuple britannique se tourna vers un nouveau leader : Winston Churchill.', 'le homme a dit ?? cause de un accident ?? cause pour un bon pour sa no??l.', 0, 12, 1], ['Qui ??tes-vous suppos??e ??tre ?', 'qui ??tes-vous toujours ?', 0, 4, 2], ["Il y avait de nombreuses choses que nous n'avions simplement pas le temps de faire.", 'il y a des choses que nous avons tous ce que nous avons besoin de faire.', 0, 15, 2], ["J'esp??re qu'ils font bon voyage.", "j'esp??re que nous aurons un bon voyage.", 0, 5, 1], ['Elle sait certainement cela.', 'elle sait le revoir.', 0, 4, 2], ["Tom n'a pas r??pondu ?? la question.", "tom n'a pas de la question.", 0, 6, 3], ["Vous ne m'entendez pas me plaindre, n'est-ce pas ?", "vous ne me voulez pas quoi vous dit n'est-ce pas ?", 0, 9, 2], ["Est-ce que c'est quelque chose de tr??s grave ?", "est-ce que c'est quelque chose de tr??s arriv??.", 0, 8, 7], ['Il tient toujours ses promesses.', 'il porte toujours ses promesses.', 0, 5, 4], ['Ma m??re ne se plaint presque jamais.', 'ma m??re ne me plaint presque jamais.', 0, 7, 6], ['Je me suis figur?? que vous ne veniez pas.', 'je pensais que vous seriez ne sont pas partir.', 0, 9, 1], ["C'est pourquoi je quitte le poste.", "c'est pourquoi je me souviens le poste.", 0, 6, 3], ['Ce sont des contrefa??ons.', 'ce sont des sont', 0, 4, 3], ["Vous n'??tes jamais sujet au doute, n'est-ce pas ?", "tu as peut-??tre chose ?? n'est-ce pas ?", 0, 8, 0], ['?? qui allez-vous offrir des cadeaux de No??l cette ann??e ?', '?? qui vous offrir un prix de no??l ?', 0, 9, 5], ["Tu n'as pas besoin de r??pondre ?? cette question.", "vous n'??tes pas besoin de r??pondre ?? cette question.", 0, 9, 7], ["J'??tais d??j?? au lit.", "j'??tais d??j?? au lit.", 1, 4, 4], ['On vous avait dit de rester sur le bateau.', 'on vous dit dit les yeux de rester ?', 0, 9, 3], ["Laisse-moi m'en charger !", "laisse-moi m'en m'en !", 0, 4, 3], ['Les Japonais mangent du riz au moins une fois par jour.', 'les enfants ont une fois par semaine.', 0, 7, 1], ['Cet endroit est fantastique.', 'cet endroit est parfait.', 0, 4, 3], ["Il s'est cogn?? la t??te contre l'??tag??re.", 'il a perdu la t??te sur la t??te au t??te de main mon p??re.', 0, 7, 3], ['Il appuya sur le bouton et attendit.', 'il a demand?? les bouton de c??t?? de la maison est vous.', 0, 7, 2], ['Quel est son but ?', 'quel est son est ?', 0, 5, 4], ['Tout le monde a d?? se mettre ?? travailler.', 'tout le monde a d?? faire les choses ?? travailler.', 0, 9, 5], ['Tu dois ?? ceux qui d??pendent de toi de le faire.', "vous n'??tes pas sujet ?? tout le faire pour ce qu'il vous ai faire.", 0, 11, 0], ["Elle rejeta notre proposition de l'aider.", 'elle a notre plan.', 0, 4, 2], ["??a nous est ??gal, si tu prends une photo depuis l'ext??rieur.", '??a ne peut nous nous aller sur une photo avec cette ville de toi.', 0, 11, 3], ["J'ob??is toujours aux r??gles.", 'je dois les r??gles.', 0, 4, 1], ['Veuillez mettre vos chaussures.', 'veuillez mettre vos chaussures.', 1, 4, 4], ['Tom perdit la course.', 'tom a la premi??re fois que marie.', 0, 4, 2], ['Je sais que vous ??tes intelligentes.', 'je sais que tu es vraiment est occup??.', 0, 6, 3], ['Tu ne peux pas rester l??-dedans toute la journ??e.', 'vous ne pouvez pas rester toute la nuit ?', 0, 9, 3], ["J'en ai assez d'??tre trait?? comme un enfant.", "j'en ai assez d'??tre un enfant.", 0, 6, 4], ["J'y ai r??fl??chi.", 'je suis en train de faire ??a.', 0, 3, 0], ['Nous sommes tous fiers de toi.', 'nous sommes tous vous sommes sur toi.', 0, 6, 3], ['Je suis le plus jeune ici.', 'je suis la plus ici.', 0, 5, 3], ["Il t'attend chez nous.", "il t'attend chez toi.", 0, 4, 3], ["Elle s'est disput??e avec lui au sujet de l'??ducation de leurs enfants.", 'elle lui a accept?? de sa vie.', 0, 7, 1], ['La serrure est cass??e.', 'la pi??ce est cass??e.', 0, 4, 3], ['Je pense que Tom a besoin de quelque chose.', 'je pense que tom a besoin de quelque chose.', 1, 9, 9], ['Sonne la cloche.', 'o?? le tom est le mien.', 0, 3, 0], ["Tom a dit qu'il partirait lundi.", "tom a dit qu'il se veut ?? la question.", 0, 6, 4], ['Elle est aussi grande que toi.', 'elle est aussi grande que toi.', 1, 6, 6], ['Je ne suis pas si pr??occup??.', 'je ne suis pas si ??a que ??a.', 0, 6, 5], ['Vous ??tes tr??s curieuses.', 'vous ??tes fort occup??.', 0, 4, 2], ['Les hommes et les femmes ont besoin les uns des autres.', 'les hommes et les hommes ont besoin des autres.', 0, 9, 6], ["Regarde l'image en haut de la page.", "regarde l'image en haut de la pi??ce de main ?", 0, 7, 6], ['Me voici.', 'me voici.', 1, 2, 2], ['La t??l?? est rest??e allum??e toute la nuit.', 'la maison est rest??e toute la nuit.', 0, 7, 3], ['Pouvez-vous vous en sortir saines et sauves ?', 'pouvez-vous vous en sortir saines et sauves ?', 1, 8, 8], ['Tom a traduit la lettre en fran??ais pour moi.', 'tom a demand?? la lettre pour fran??ais pour moi.', 0, 9, 7], ["Il est plus que probable qu'il vienne.", "il est plus que probable qu'il y aller.", 0, 7, 6], ['Il tint sa promesse et aida ses fr??res.', 'il ses doigts de sa banque.', 0, 6, 1], ["Vous n'??tes plus le bienvenu chez moi.", "vous n'??tes pas la m??me chez moi.", 0, 7, 4], ["J'appr??cie tout ce que tu peux faire.", 'je pense que vous devriez quelque chose.', 0, 7, 0], ["Pourquoi ??tes-vous en train d'attendre ?", 'pourquoi ??tes-vous en train de train plus que ?', 0, 6, 4], ["Ils s'accord??rent pour l'??lire pr??sident.", "ils s'accord??rent pour son fois ?", 0, 5, 3], ["J'ai enlev?? la virgule.", "j'ai enlev?? la question.", 0, 4, 3], ['Ce chapitre sera consacr?? aux concepts de la g??om??trie.', "ce sera du bruit aux consacr?? aux aux que j'ai une de deux.", 0, 9, 2], ['Que fait ce monsieur, dans la vie ?', 'que ce fait une seule ?', 0, 6, 1], ['Il a fait beaucoup de choses pour les pauvres.', 'il a beaucoup de choses pour les choses pour les choses de monde.', 0, 9, 2], ['Il a fait comme je le lui ai dit.', 'il a dit que je le lui dit.', 0, 8, 5], ["Prends note de ce qu'il dit, s'il te plait.", "veuillez donner un dire de papier s'il te plait.", 0, 9, 3], ['Il refuse de me dire ce qui est arriv??.', "il ne me dit que ce n'est pas de la maison.", 0, 9, 2], ['??tes-vous satisfait du service?', '??tes-vous satisfait du caf?? ?', 0, 4, 3], ['Tom passa trois ans en prison.', 'tom passa trois ans en prison.', 1, 6, 6], ["Qui est de garde aujourd'hui\u202f?", 'qui est de gens se gens ?', 0, 6, 3], ['Est-ce que je peux faire ??a ici\xa0?', 'est-ce que je peux faire ??a le faire\xa0?', 0, 8, 6], ['Quelle est la montagne la plus haute au monde ?', 'quelle est la montagne la plus de la plus de la plus ?', 0, 10, 6], ["Six ans ont pass?? depuis que j'ai commenc?? ?? ??tudier l'anglais.", 'six ans ont pass?? ?? trois ans depuis ??tudier un ??tais', 0, 11, 4], ['Tom ne peut plus nous faire de mal.', 'tom ne peut pas nous faire de mal.', 0, 8, 7], ["Le gar??on s'est presque noy??.", "le gar??on a ??t?? presque par ??t?? par l'??cole.", 0, 5, 2], ['Tom obtient certainement un beau son de ce vieux basson.', "tom veut certainement un beau grand de temps qu'il nouvelle semble fier de marie.", 0, 10, 5], ["C'est toi qui devrais payer la facture.", "c'est la seule qui vous plait.", 0, 6, 1], ["Ce n'est pas o?? je me rends.", "ce n'est pas que je suis en train de main ?", 0, 7, 4], ['Il est devenu riche par un dur labeur.', 'il est devenu vers faire depuis son monde.', 0, 8, 3], ["Je n'ai aucune id??e de comment ??a marche.", "je n'ai aucune id??e de comment ??a ??a.", 0, 8, 7], ["Tom n'a pas de casier judiciaire.", "tom n'a pas de casier ses autres.", 0, 6, 5], ['Bient??t vos efforts seront r??compens??s.', 'ton efforts vos efforts', 0, 4, 0], ["Viens avec nous, s'il te plait.", "veuillez venir avec nous, s'il vous plait.", 0, 6, 0], ["J'en ai fini avec toi.", 'je suis avec toi.', 0, 4, 0], ["C'est un champion du monde en puissance.", "c'est un homme du monde en puissance.", 0, 7, 6]]



```python
df = pd.DataFrame(test)
print("values are equal to 1 if the prediction and the expected translation match exactly, else it's 0")
print(df[2].value_counts())
print("Number of correct words at correct indexes:",df[4].sum())
print("Total number of words:",df[3].sum())
print("Percentage of correct words over a sample of 500 sentences:",(df[4].sum()/df[3].sum()))
```

    values are equal to 1 if the prediction and the expected translation match exactly, else it's 0
    0    437
    1     63
    Name: 2, dtype: int64
    Number of correct words at correct indexes: 1568
    Total number of words: 3237
    Percentage of correct words over a sample of 500 sentences: 0.48439913500154463

