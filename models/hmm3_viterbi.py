

class Viterbi():

    def __init__(self, trans_p, emit_p, label2index, word2index):
        """
        Initialize the Model. Expects log probabilities!

        Args:
            trans_p:        numpy array with label transition probabilities: P(s|s-2,s-1)
            emit_p:         numpy array with emission probabilities: P(w|s)
            label2index:    dict mapping labels into matrix indices
            word2index:     dict mapping tokens into matrix indices
        """

        self.trans_p = trans_p
        self.emit_p = emit_p
        self.label2index = label2index
        self.word2index = word2index

    
    def search(self, x, states):
        """
        Search for label sequence with highest probability.

        Args:
            x:       string
            states:  list of candidate labels

        Returns:
            Tuple with two elements:
            - list: sequence of labels predicted
            - prob: probability of prediction

        """
        # Possible transition states at state i
        def S(i):
            return states if i>=0 else ['<START>']

        # Transition probability
        def Q(a, b, c):
            try:
                i, j, k = [self.label2index[s] for s in (a,b,c)]
                return self.trans_p[i][j][k]
            except Exception:
                return -50

        # Emission probability
        def E(w, s):
            try:
                w = self.word2index[w]
                s = self.label2index[s]
                return self.emit_p[w][s]
            except Exception:
                return -50

        # Path (highest sequence probability for each tag)
        P = {}
        # History of probabilities
        H = [{}]

        # Initialize with starting probabilities (t=0)
        t = 0
        for s in S(t):
            H[t][s] = {}
            H[t][s]['<START>'] = Q('<START>', '<START>', s) + E(x[t], s)
            P[s] = [s]

        # Run Viterbi for t > 0
        for t in range(1, len(x)):
            H.append({})
            newP = {}

            for s in S(t):
                # read s_1, s_2 as s-1, s-2
                H[t][s] = {}
                new_max = []
                for s_1 in S(t-1):
                    (max_p, max_s_2) = max( (H[t-1][s_1][s_2] + Q(s_2, s_1, s) + E(x[t], s), s_2) for s_2 in S(t-2) )
                    H[t][s][s_1] = max_p
                    new_max.append( (max_p, max_s_2, s_1) )

                _, _, max_s_1 = max(new_max)
                newP[s] = P[max_s_1] + [s]
            P = newP

        # Final step
        t = len(x)
        (max_p, max_s) = max( (H[t-1][s_1][s_2] + Q(s_2, s_1, '<STOP>'), s_1 ) for s_1 in S(t-1) for s_2 in S(t-2) )


        return P[max_s], max_p
