from pymongo import MongoClient
from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
import nltk

client = MongoClient("localhost", 27017)

db = client['Assignment4']
termsCol = db['terms']
docsCol = db['documents']

def run_query(query):
    # Get vocab list
    terms = termsCol.distinct("term")

    # Generate unigrams, bigrams and trigrams for vocab words within query
    unigrams = [part.lower() for part in query.split(" ") if part.lower() in terms]
    bigrams = [f'{a} {b}' for a, b in list(nltk.bigrams(unigrams))]
    trigrams = [f'{a} {b} {c}' for a, b, c in list(nltk.trigrams(unigrams))]
    
    # Combine parts & fetch relevant term info
    parts = unigrams + bigrams + trigrams
    termInfo = list(termsCol.find({ "term": { "$in": parts }}))

    # Return empty result set if no matching query terms found
    if(len(termInfo) == 0):
        return []

    # Define query vector
    q0 = []
    # Define array to store relevant document IDs
    docIds = []

    # Iterate terms
    for term in termInfo:
        idf = float(term['idf']) # Fetch IDF value
        q0.append((parts.count(term['term'])/len(parts)) * idf) # Append tf-idf value to query vector
        for doc in term['docs']:
            if doc not in docIds:
                docIds.append(ObjectId(doc)) # Append relevant document to document IDs

    # Fetch relevant document information
    docs = list(docsCol.find({ "_id": { "$in": docIds}}))

    # Generate document tf-idf vectors only considering non-zero values
    docV = [[doc['tfidf'][int(term['pos'])] for term in termInfo] for doc in docs]

    # Perform cosine similarity and sort descending
    results = cosine_similarity([q0] + docV)[0].tolist()
    results = sorted(list(enumerate(results[1:])), key=lambda tup: tup[1], reverse=True)

    # Return document info and similarity score
    out = []
    for idx, similarity in results:
        out.append((docs[idx]['content'], similarity*100))

    return out

# Prompt user for search
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