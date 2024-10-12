class ReRank:


    def reRank(self, topics, nDocs, opensearch):



        for topic in topics:

            bm25rank = opensearch.search_body(topics[topic], numDocs=nDocs)['_id']

            for doc in bm25rank:
                termVec = opensearch.termvectors_JSON(doc_id=doc)
                print(termVec)

                terms = termVec['term_vectors']['contents']['terms']
                sum_ttf = termVec['term_vectors']['contents']['field_statistics']['sum_ttf']
                print('=========== sum_ttf')

                d = sum(terms[term]['term_freq'] for term in terms)
                print('=========== d1: ')
                print(d)

                miu = 0.1 * d
                print('=========== miu com d3: ')
                print(miu)

                res = 1
                for term in terms:
                    term_freq = terms[term]['term_freq']





        return 0
