import math

import pandas as pd


class ReRank:


    def reRank(self, topic, opensearch):

        res = opensearch.search_body(topic, numDocs=100)
        # print(res)

        queryTerms = opensearch.query_terms(analyzer="standard", query=topic).split()
        # print(queryTerms)

        topDocs = res[:50]['_id'].tolist()
        # topDocs = res['_id'].tolist()

        col_doc_freq, col_sum_doc_freq, col_sum_ttf, col_docs_term_vectors = opensearch.multi_doc_term_vectors(
            doc_ids=topDocs)

        ranked_results = []

        dm = sum(doc[terms][term][0] for term in col_docs_term_vectors)

        # print(
        #     f'================================== Doc_freq: {col_doc_freq} Sum_doc_freq: {col_sum_doc_freq} Sum_ttf: {col_sum_ttf}')

        for doc, terms in col_docs_term_vectors.items():
            # print(doc)
            # print(terms)


            d = sum(terms[term][0] for term in terms)
            miu = 0.1 * d
            # miu = 0.1 * (col_sum_ttf/d)

            res = 0
            for term in queryTerms:
                # print('TERM: ' + str(term))

                if term not in terms:
                    # print('NÃO ESTÁ -------------> ' + str(term))

                    found = False
                    for col_docs, col_terms in col_docs_term_vectors.items():
                        if term in col_terms:
                            found = True
                            ttf = col_terms[term][2]
                            # print(col_terms[term])
                            # print('ttf' + str(ttf))

                            term_freq = 0
                            prob = ttf / col_sum_ttf

                            numerator = term_freq + miu * prob
                            denominator = miu + d

                            func = numerator / denominator

                            res += math.log(func)
                            # LMDrank[doc] = math.exp(res)
                            break

                    # if not found:
                    # print('NÃO ESTÁ GERAL -------------> ' + str(term))

                else:
                    # print('ESTÁ: ' + str(term))
                    # print(terms[term])

                    term_freq = terms[term][0]
                    prob = terms[term][2] / col_sum_ttf

                    numerator = term_freq + miu * prob
                    denominator = miu + d

                    func = numerator / denominator

                    res += math.log(func)
                    # LMDrank[doc] = math.exp(res)

            ranked_results.append({
                '_id': doc,
                '_score': res,
                '_source.doc': doc
            })

        result_df = pd.DataFrame(ranked_results).sort_values(by='_score', ascending=False)

        result_df.reset_index(drop=True, inplace=True)

        return result_df



