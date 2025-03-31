import pandas as pd
from nltk.corpus import stopwords
import emoji
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import contractions
import enchant
from enchant.checker import SpellChecker
import re


from gensim.models import Word2Vec
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet



def correct_spelling(text):
    """Corrige les fautes d'orthographe avec enchant."""
    checker = SpellChecker("en_US")
    checker.set_text(text)
    for error in checker:
        if error.suggest():  # S'il y a des suggestions
            error.replace(error.suggest()[0])  # Remplace par la première suggestion
    return checker.get_text()


def lemmatize_text(text):
    """Pipeline de prétraitement : Correction → Contractions → Lemmatisation."""
    # Charger le modèle de langue anglaise de spaCy
    nlp = spacy.load('en_core_web_sm')

    # Étape 1: Correction orthographique
    text = correct_spelling(text)

    # Étape 2: Gestion des contractions (ex: "that's" → "that is")
    text = contractions.fix(text)

    # Étape 3: Remplacement manuel des 's résiduels (optionnel)
    text = text.replace("'s", " be")

    # Étape 4: Lemmatisation avec spaCy
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)

def make_dataset(dataset = pd.read_csv('scitweets_export.tsv', sep='\t'),
                 with_affirmation=True,):
    dataset['text'] = dataset['text'].apply(lambda x: x.lower())


    stop_words = set(stopwords.words('english'))
    work_to_keep = ['no', 'not', 'nor', 'too', 'very', 'against', 'but', 'don', 'don\'t', 'ain', 'aren', 'aren\'t',
                    'couldn', 'couldn\'t', 'didn', 'didn\'t', 'doesn', 'doesn\'t', 'hadn', 'hadn\'t', 'hasn', 'hasn\'t',
                    'haven', 'haven\'t', 'isn', 'isn\'t', 'mightn', 'mightn\'t', 'mustn', 'mustn\'t', 'needn',
                    'needn\'t', 'shan', 'shan\'t', 'shouldn', 'shouldn\'t', 'wasn', 'wasn\'t', 'weren', 'weren\'t',
                    'won', 'won\'t', 'wouldn', 'wouldn\'t']

    if with_affirmation:
        for word in work_to_keep:
            stop_words.remove(word)

    dataset['text'] = dataset['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    for index, line in dataset.iterrows():
        splitted = line['text'].split(' ')
        new_text = []
        for word in splitted:
            if 'http' in word or 'www' in word:
                continue
            new_text.append(word)
        dataset.at[index, 'text'] = ' '.join(new_text)

    # Parcourir les indices et modifier directement les valeurs dans le DataFrame
    for index, line in dataset.iterrows():
        dataset.at[index, 'text'] = emoji.demojize(line['text'])

    for index, line in dataset.iterrows():
        splitted = line['text'].split(' ')
        new_text = []
        for word in splitted:
            new_word = ""
            for l in word:
                if l.isalpha() or l in ['#', "'", '?', '!']:
                    new_word += l
                else:
                    new_word += ' '
            new_text.append(new_word)
        dataset.at[index, 'text'] = ' '.join(new_text)

    for index, line in dataset.iterrows():
        dataset.at[index, 'text'] = ' '.join(line['text'].split())

    # Appliquer la lemmatisation à la colonne 'text'
    dataset['text_lemmatized'] = dataset['text'].apply(lemmatize_text)

    return dataset

def dataset_cirian(df = pd.read_csv('scitweets_export.tsv', sep='\t')):
    text = []
    all_text = list(df["text"])
    for t in all_text:
        a = re.sub("[\.,/\\@:;\"\'0-9“”’\[\]]", "", t)
        a = re.sub("http\S*", "", a)  # Enlever les liens
        a = re.sub("#\S*", "", a)  # Enlever les tags
        a = a.lower()
        text.append(a)

    text_sep = [word_tokenize(t) for t in text]

    def get_wordnet_pos(tag):
        if tag.startswith('J'):  # Adjectif
            return wordnet.ADJ
        elif tag.startswith('V'):  # Verbe
            return wordnet.VERB
        elif tag.startswith('N'):  # Nom
            return wordnet.NOUN
        elif tag.startswith('R'):  # Adverbe
            return wordnet.ADV
        else:
            return wordnet.NOUN

    lemmatizer = WordNetLemmatizer()
    refined_tweets = []
    for tweet in text_sep:
        pos_tags = pos_tag(tweet)
        lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
        refined_tweets.append(lemmatized_words)

    valid_words = []

    d = enchant.Dict("en_US")

    # Trouver tous les mots qui sont dans la dictionnaiore anglais
    for sentence in refined_tweets:
        for w in sentence:
            if w not in valid_words:
                if d.check(w):
                    valid_words.append(w)
                if d.check(w.capitalize()) and w not in valid_words:
                    valid_words.append(w)

    total_words = []
    for sentence in refined_tweets:
        for w in sentence:
            if w not in total_words:
                total_words.append(w)

    invalid_words = []
    for sentence in refined_tweets:
        for w in sentence:
            if w not in invalid_words:
                if w not in valid_words:
                    invalid_words.append(w)

    df['text'] = text
    df['text_lemmatized'] = [' '.join(t) for t in refined_tweets]
    return df


if __name__ == "__main__":
    dataset = pd.read_csv('scitweets_export.tsv', sep='\t')
    #make_dataset(dataset).to_csv('scitweets_transformed.tsv', sep='\t', index=False)
    #make_dataset(dataset, with_affirmation=False).to_csv('scitweets_transformed_without_affirmation.tsv', sep='\t', index=False)
    #dataset_cirian(dataset).to_csv('scitweets_transformed_cirian.tsv', sep='\t', index=False)
    print("Done !")