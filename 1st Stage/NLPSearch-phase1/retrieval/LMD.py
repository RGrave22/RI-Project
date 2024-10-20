import matplotlib.pyplot as plt
import numpy as np
from retrieval_methods.LMD import LMD_Retrieval
from textwrap import wrap

class LMD:
    def __init__(self,opensearch, test_bed, conv_topics, selected_topics):
        self.opensearch = opensearch
        self.test_bed = test_bed
        self.conv_topics = conv_topics
        self.selected_topics = selected_topics

    def run(self,fig_subplot_size):
        fig = plt.figure(1, figsize= [12,9])
        fig.subplots_adjust(hspace = 0.4)
        fig.show()
        fig.suptitle("PR-curves per conversation", y=0.95)

        _ndcg5_acc = np.zeros((len(self.selected_topics),12))
        _p10_acc = np.zeros((len(self.selected_topics),12))
        _recall = 0
        _p10 = 0
        _ndcg5 = 0
        _ap = 0
        _full_mean_prec_rec = []

        verbose = False
        _nturns = 0
        _conv_count = 0

        lmd = LMD_Retrieval

        for topic in self.conv_topics:
            conv_id = topic['number']
            title = topic['title']

            if conv_id not in self.selected_topics:
                continue

            if verbose:
                print(title)

            _precision_11point = []
            _turn_count = -1
            for turn in topic['turn']:
                _turn_count = _turn_count + 1
                turn_id = turn['number']
                utterance = turn['raw_utterance']
                topic_turn_id = '%d_%d'% (conv_id, turn_id)
                
                result = lmd.retrieve_fun(self.opensearch,utterance)

                num_rel = self.test_bed.num_rel(topic_turn_id)

                if num_rel == 0:
                    _p10_acc[_conv_count, _turn_count] = 0
                    _ndcg5_acc[_conv_count, _turn_count] = 0
                    continue

                if np.size(result) == 0:
                    continue

                if verbose:
                    print(topic_turn_id, self.opensearch.query_terms(utterance,'standard'))

                [p10, recall, ap, ndcg5] = self.test_bed.eval(result[['_id','_score','_source.doc']], topic_turn_id, verbose = False)
                _p10_acc[_conv_count, _turn_count] = p10
                _ndcg5_acc[_conv_count, _turn_count] = ndcg5
                _recall = _recall + recall
                _p10 = _p10 + p10
                _ndcg5 = _ndcg5 + ndcg5
                _ap = _ap + ap
                
                [precision_11point, recall_11point, total_relv_ret] = self.test_bed.evalPR(result[['_id','_score', '_source.doc']], topic_turn_id)
                _precision_11point.append(precision_11point)

                _nturns = _nturns + 1

            mean_prec_rec = np.asarray(_precision_11point ).mean(axis=0)
            var_prec_rec = np.asarray(_precision_11point ).var(axis=0)
            _full_mean_prec_rec.append(mean_prec_rec)

            recall_11point = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ax = fig.add_subplot(fig_subplot_size[0],fig_subplot_size[1],_conv_count+1)
            ax.set_ylim([0.0,1.0])
            if len(title) > 30:
                ax.set_title("\n".join(wrap(title, 28)), fontsize=10)
            else:
                ax.set_title(title, fontsize=10)
            ax.fill_between(recall_11point, mean_prec_rec - var_prec_rec, mean_prec_rec + var_prec_rec, alpha = 0.1)
            ax.set_xticks(np.arange(0,1.1,0.2))
            ax.plot(recall_11point, mean_prec_rec)
            _conv_count = _conv_count + 1
        
        _p10 = _p10/_nturns
        _recall = _recall/_nturns
        _ndcg5 = _ndcg5/_nturns
        _ap = _ap/_nturns

        fig_2 = plt.figure(2, figsize= [8,3])
        fig_2.subplots_adjust(wspace = 0.3)

        mean_p10_per_turn = np.mean(_p10_acc,axis = 0)
        var_p10_per_turn = np.var(_p10_acc,axis = 0)
        ax1 = fig_2.add_subplot(1,2,1)
        ax1.plot(np.arange(1,9), mean_p10_per_turn[0:8])
        ax1.set_xticks(np.arange(1,9))
        ax1.fill_between(np.arange(1,9), mean_p10_per_turn[0:8]-var_p10_per_turn[0:8], mean_p10_per_turn[0:8]+var_p10_per_turn[0:8], alpha = 0.1)
        ax1.set_ylim([0.0,1.0])
        ax1.set_xlabel("Conversation turn")
        ax1.set_ylabel("Precision @ 10")
        ax1.set_title("Mean P@10 per turn")

        ax2 = fig_2.add_subplot(1,2,2)
        full_mean_prec_rec = np.asarray(_full_mean_prec_rec ).mean(axis=0)
        full_var_prec_rec = np.asarray(_full_mean_prec_rec ).var(axis=0)
        ax2.plot(recall_11point, full_mean_prec_rec, color='red')
        ax2.fill_between(recall_11point, full_mean_prec_rec - full_var_prec_rec, full_mean_prec_rec + full_var_prec_rec, alpha = 0.1)
        ax2.set_ylim([0.0,1.0])
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Overall PR-curve")

        print()
        print('P10=', _p10, '  Recall=', _recall, '  NDCG@5=', _ndcg5, '  AP=', _ap)
        print()

        fig.savefig('results/lmd_pr_curves_per_conv.png', format='png', dpi=600, transparent=True)
        fig_2.savefig('results/lmd_meanp10_and_overall_pr_curve.png', format='png', dpi=600, transparent=True, bbox_inches='tight')
    