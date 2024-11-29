from sklearn.feature_extraction.text import TfidfVectorizer
import string, re
from pymongo import MongoClient

client = MongoClient("localhost", 27017)
db = client['Assignment4']
termsCol = db['terms']
docsCol = db['documents']

# Define docs
originalDocs = ['After the medication, headache and nausea were reported by the patient.',
                'The patient reported nausea and dizziness caused by the medication.',
                'Headache and dizziness are common effects of this medication.',
                'The medication caused a headache and nausea, but no dizziness was reported.']

# Convert to lowercase and strip punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
docs = [regex.sub('', doc.lower()) for doc in originalDocs]

# Define array to store generated document IDs
docIds = []

# Fit documents
vectorizer = TfidfVectorizer(analyzer= 'word', ngram_range=(1, 3)) # Generate unigrams, bigrams and trigrams
vectorizer.fit(docs)

# Transform documents
docsv = vectorizer.transform(docs).toarray()

# Store IDF values
idf = vectorizer.idf_.tolist()

# Push docs to MongoDB
index = 0
for doc in originalDocs:
    inserted = docsCol.insert_one({
        "content": doc,
        "tfidf": docsv[index].tolist()
    })
    docIds.append(inserted.inserted_id)
    index+=1

# Push terms to MongoDB
for term, idx in vectorizer.vocabulary_.items():
    docList = []
    index = 0
    for doc in docs:
        if term in doc:
            docList.append(docIds[index])
        index+=1
    termsCol.update_one({"term": term}, { "$set": {
        "term": term,
        "pos": idx,
        "idf": idf[idx],
        "docs": docList
    }}, upsert=True)