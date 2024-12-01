import math

import pandas as pd


class ReRank:


    def reRank(self, topic, opensearch, numDocs):

        res = opensearch.search_body(topic, numDocs)

        queryTerms = opensearch.query_terms(analyzer="standard", query=topic).split()

        # Limitar a 50 docs para não dar erro nos graficos
        topDocs = res[:numDocs]['_id'].tolist()

        #Com os 100 docs da erro
        #topDocs = res['_id'].tolist()
        print(topDocs)
        print(len(topDocs))

        col_doc_freq, col_sum_doc_freq, col_sum_ttf, col_docs_term_vectors = opensearch.multi_doc_term_vectors(
            doc_ids=topDocs)

        ranked_results = []


        dm = 0
        for doc, terms in col_docs_term_vectors.items():
            d = sum(terms[term][0] for term in terms)
            dm += d

        for doc, terms in col_docs_term_vectors.items():

            d = sum(terms[term][0] for term in terms)
            miu = 0.1 * dm/100

            res = 0
            for term in queryTerms:

                if term not in terms:

                    for col_docs, col_terms in col_docs_term_vectors.items():
                        if term in col_terms:

                            ttf = col_terms[term][2]

                            term_freq = 0
                            prob = ttf / col_sum_ttf

                            numerator = term_freq + miu * prob
                            denominator = miu + d

                            func = numerator / denominator

                            res += math.log(func)

                            break



                else:

                    term_freq = terms[term][0]
                    prob = terms[term][2] / col_sum_ttf

                    numerator = term_freq + miu * prob
                    denominator = miu + d

                    func = numerator / denominator

                    res += math.log(func)

            ranked_results.append({
                '_id': doc,
                '_score': res,
                '_source.doc': doc
            })

        result_df = pd.DataFrame(ranked_results).sort_values(by='_score', ascending=False)

        result_df.reset_index(drop=True, inplace=True)

        return result_df



