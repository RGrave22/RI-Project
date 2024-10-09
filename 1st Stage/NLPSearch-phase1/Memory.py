
class Memory:

    def __init__(self):
        self.memory = {}

    def storeMem(self, topics, nDocs, nTop, opensearch):

        for topic in topics:
            print("=================" + topic + "================")
            results = opensearch.search_body(topics[topic], numDocs=nTop)
            #print(results[:nTop]["_id"])
            title = topics[topic]

            top = [results["_id"]]
            print(title)
            self.memory[topic] = (title, top)

        #print(self.memory)

        return self.memory

