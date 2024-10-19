import pandas as pd
import ReRank

class rank_with_history:


    
    def rankWithHistory(self, first_utterance, turn, opensearch):
        rr = ReRank.ReRank()   
        query = first_utterance + " " + turn['raw_utterance']
        
        print("\nQUERY: " + query + "\n")

        #Normal BM25
        res = opensearch.search_body(query, numDocs=100)

        #LMD
        #res = rr.reRank(topic = query, opensearch = opensearch)

        return res

