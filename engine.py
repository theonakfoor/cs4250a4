from pymongo import MongoClient
from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
import nltk

client = MongoClient("localhost", 27017)

db = client['Assignment4']
termsCol = db['terms']
docsCol = db['documents']

def run_query(query):
    terms = termsCol.distinct("term")

    unigrams = [part.lower() for part in query.split(" ") if part.lower() in terms]
    bigrams = [f'{a} {b}' for a, b in list(nltk.bigrams(unigrams))]
    trigrams = [f'{a} {b} {c}' for a, b, c in list(nltk.trigrams(unigrams))]
    
    parts = unigrams + bigrams + trigrams

    termInfo = termsCol.find({ "term": { "$in": parts }})

    q0 = [0 for _ in terms]
    docIds = []

    for term in list(termInfo):
        idf = float(term['idf'])
        if idf > 0:
            q0[int(term['pos'])] = (parts.count(term['term'])/len(parts)) * idf
        for doc in term['docs']:
            if doc not in docIds:
                docIds.append(ObjectId(doc))

    docs = list(docsCol.find({ "_id": { "$in": docIds}}))
    docV = [doc['tfidf'] for doc in docs]

    results = cosine_similarity([q0] + docV)[0].tolist()
    results = sorted(list(enumerate(results[1:])), key=lambda tup: tup[1], reverse=True)

    out = []
    for idx, similarity in results:
        out.append((docs[idx]['content'], similarity*100))

    return out

while True:
    query = input("Enter a search query (Enter 'q' to Quit): ")
    if query.lower() != 'q':
        results = run_query(query)
        print(f"\nResults for '{query}':\n")
        if(len(results) == 0): print("No results found.")
        for content, similarity in results:
            print(f'{content} -- {round(similarity, 2)}%')
        print()
    else:
        break