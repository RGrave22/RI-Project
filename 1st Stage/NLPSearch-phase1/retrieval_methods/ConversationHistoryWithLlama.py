import RankWithHistoryLlama as rwh

rwh = rwh.rank_with_history()

class ConversationHistoryWithLlama_Retrieval:
    utt_limit = 2
    
    def __init__(self,opensearch):
        self.history_query_list = []
        self.opensearch = opensearch
        self.last_utterance = ""


    def sliding_window(self,utterance,turn):
        if(len(self.history_query_list) == 0):
            self.history_query_list.append(utterance)
            return self.opensearch.search_body(query=utterance, numDocs = 100)
        elif(len(self.history_query_list) < self.utt_limit):
            self.history_query_list.append(self.last_utterance)
            self.last_utterance = utterance
            return rwh.rankWithHistoryLlama(first_utterance= ' '.join(self.history_query_list), turn = turn, opensearch = self.opensearch)
        else:
            self.history_query_list.pop(1)
            self.history_query_list.append(self.last_utterance)
            self.last_utterance = utterance
            return rwh.rankWithHistoryLlama(first_utterance=' '.join(self.history_query_list), turn = turn, opensearch= self.opensearch)
        
    def first_utt_only(self, utterance,turn,turn_id):
        if turn_id == 1 :
            self.first_utterance = utterance
            return self.opensearch.search_body(query=utterance, numDocs = 100)
        else:
            return rwh.rankWithHistoryLlama(first_utterance= self.first_utterance, turn = turn, opensearch = self.opensearch)