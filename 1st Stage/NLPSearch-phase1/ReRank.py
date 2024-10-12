class ReRank:


    def reRank(self, topics, nDocs, opensearch, memory):

        aux = []
        # for topic in topics:
        #     bm25rank = opensearch.search_body(topics[topic], numDocs=nDocs)['_id']

        for topic, (query, doc_ids) in  memory.items():

            print(doc_ids)

            for doc_id in doc_ids:
                print('DOC_ID -> '+ str(doc_id))
            #     termVec = opensearch.termvectors_JSON(doc_id=doc_ids[doc_id]['_id'])
            #     aux.append(termVec)

        print(aux)

        # for topic in memory[topics]:
        #
        #
        #
        #     for doc in memory:
        #         termVec = opensearch.termvectors_JSON(doc_id=doc)
        #         print(termVec)
        #
        #         terms = termVec['term_vectors']['contents']['terms']
        #         sum_ttf = termVec['term_vectors']['contents']['field_statistics']['sum_ttf']
        #         print('=========== sum_ttf')
        #
        #         d = sum(terms[term]['term_freq'] for term in terms)
        #         print('=========== d1: ')
        #         print(d)
        #
        #         miu = 0.1 * d
        #         print('=========== miu com d3: ')
        #         print(miu)
        #
        #         res = 1
        #         for term in terms:
        #             term_freq = terms[term]['term_freq']




