import numpy as np
from typing import List

class CRFNumpy:

    def __init__(self, transitions, start_transitions, end_transitions):
        self.transitions = transitions
        self.start_transitions = start_transitions
        self.end_transitions = end_transitions

    def viterbi_decode(self, emissions: np.ndarray,
                        mask: np.ndarray) -> List[List[int]]:

        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = np.expand_dims(score, axis=2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = np.expand_dims(emissions[i], axis=1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            indices = np.argmax(next_score, axis=1)
            next_score = np.take_along_axis(next_score, np.expand_dims(indices, axis=1), axis=1).squeeze(axis=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = np.where(np.expand_dims(mask[i], axis=1), next_score, score)
            history.append(indices)

        history = np.stack(history, axis=0)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.astype('int').sum(axis=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            best_last_tag = score[idx].argmax(axis=0)
            best_tags = [best_last_tag]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for i, hist in enumerate(np.flip(history[:seq_ends[idx]], axis=0)):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag)

            #best_tags = np.stack(best_tags, axis=0)

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list