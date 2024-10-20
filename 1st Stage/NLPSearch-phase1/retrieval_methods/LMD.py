import ReRank as rr

rr = rr.ReRank()

class LMD_Retrieval: 
    def retrieve_fun(opensearch, utterance):
        return rr.reRank(topic=utterance,opensearch=opensearch) 