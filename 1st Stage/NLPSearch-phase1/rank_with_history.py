import pandas as pd
import ReRankLMD

class rank_with_history:


    
    def rankWithHistory(self, first_utterance, turn, opensearch):
        rr = ReRankLMD.ReRank()   
        query = first_utterance + " " + turn['raw_utterance']
        
        #Normal BM25
        res = opensearch.search_body(query, numDocs=100)

        #LMD
        #res = rr.reRank(topic = query, opensearch = opensearch)

        print("\nQUERY: " + query + "\n")

        return res

    # def rankWithHistory_title(self, topic, turn, opensearch):

    #     query = topic['title'] + " " + turn['raw_utterance']

    #     res = opensearch.search_body(query, numDocs=100)

    #     return res

