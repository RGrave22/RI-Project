class BM25_Retrieval: 
    def retrieve_fun(opensearch, utterance):
        return opensearch.search_body(query=utterance, numDocs = 100)