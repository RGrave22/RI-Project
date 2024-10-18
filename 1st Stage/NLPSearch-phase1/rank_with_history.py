import pandas as pd


class rank_with_history:



    def rankWithHistory(self, first_utterance, turn, opensearch):

        query = first_utterance + " " + turn['raw_utterance']
        
        print("\nQUERY: " + query + "\n")

        res = opensearch.search_body(query, numDocs=100)

        return res

