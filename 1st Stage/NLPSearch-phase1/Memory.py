import pandas as pd


class Memory:

    def __init__(self):
        self.memory = pd.DataFrame(columns=["Topic", "Title", "Top_Results"])
        self.list = []

    def storeMem(self, topics, nDocs, nTop, opensearch):

        for topic in topics:
            results = opensearch.search_body(topics[topic], numDocs=nDocs)

            title = topics[topic]

            top = results[:nTop]["_id"].tolist()

            self.list.append({
                "Topic": topic,
                "Title": title,
                "Top_Results": top
            })

        self.memory = pd.DataFrame(self.list)


        return self.memory

