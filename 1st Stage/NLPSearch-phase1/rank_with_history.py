import pandas as pd


class rank_with_history:



    def rankWithHistory(self, topic, turn, opensearch):

        query = topic['title'] + " " + turn['raw_utterance']

        res = opensearch.search_body(query, numDocs=100)

        return res

