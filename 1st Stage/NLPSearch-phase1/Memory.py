
class Memory:

    def __init__(self):
        self.memory = {}

    def storeMem(self, topics, nDocs, nTop, opensearch):

        for topic in topics:
            print("=================" + topic + "================")
            results = opensearch.search_body(topics[topic], numDocs=nDocs)
            print(results[:nTop]["_id"])
            title = topics[topic]

            top = results[:nTop]["_id"].tolist()
            print('TOP -->' + str(top))
            print(title)
            self.memory[topic] = (title, top)

        #print(self.memory)

        return self.memory

