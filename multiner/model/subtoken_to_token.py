import torch

class SubtokenToToken(torch.nn.Module):

    def forward(self, units: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Assemble token level units from subtoken level units
        Args:
            units: torch.Tensor of shape [batch_size, SUBTOKEN_seq_length, n_features]
            mask: mask of token beginnings. For example: for tokens
                    [[``[CLS]`` ``My``, ``capybara``, ``[SEP]``],
                    [``[CLS]`` ``Your``, ``aar``, ``##dvark``, ``is``, ``awesome``, ``[SEP]``]]
                the mask will be
                    [[0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1, 1, 0]]
        Returns:
            word_level_units: Units assembled from ones in the mask. For the
                example above this units will correspond to the following
                    [[``My``, ``capybara``],
                    [``Your`, ``aar``, ``is``, ``awesome``,]]
                the shape of this tensor will be [batch_size, TOKEN_seq_length, n_features]
        """
        device = units.device
        nf_int = units.size()[-1]
        batch_size = units.size()[0]

        # number of TOKENS in each sentence
        token_seq_lengths = torch.sum(mask, 1).to(torch.int64)
        # for a matrix m =
        # [[1, 1, 1],
        #  [0, 1, 1],
        #  [1, 0, 0]]
        # it will be
        # [3, 2, 1]

        n_words = torch.sum(token_seq_lengths)
        # n_words -> 6

        max_token_seq_len = torch.max(token_seq_lengths)
        # max_token_seq_len -> 3

        idxs = torch.stack(torch.nonzero(mask, as_tuple=True), dim=1)
        # for the matrix mentioned above
        # tf.where(mask) ->
        # [[0, 0],
        #  [0, 1]
        #  [0, 2],
        #  [1, 1],
        #  [1, 2]
        #  [2, 0]]

        sample_ids_in_batch = torch.nn.functional.pad(input=idxs[:, 0], pad=[1, 0])
        # for indices
        # [[0, 0],
        #  [0, 1]
        #  [0, 2],
        #  [1, 1],
        #  [1, 2],
        #  [2, 0]]
        # it is
        # [0, 0, 0, 0, 1, 1, 2]
        # padding is for computing change from one sample to another in the batch

        a = (~torch.eq(sample_ids_in_batch[1:], sample_ids_in_batch[:-1])).to(torch.int64)
        # for the example above the result of this statement equals
        # [0, 0, 0, 1, 0, 1]
        # so data samples begin in 3rd and 5th positions (the indexes of ones)

        # transforming sample start masks to the sample starts themselves
        q = a * torch.arange(n_words, device=device).to(torch.int64)
        # [0, 0, 0, 3, 0, 5]
        count_to_substract = torch.nn.functional.pad(torch.masked_select(q, q.to(torch.bool)), [1, 0])
        # [0, 3, 5]

        new_word_indices = torch.arange(n_words, device=device).to(torch.int64) - count_to_substract[torch.cumsum(a, 0)]
        # tf.range(n_words) -> [0, 1, 2, 3, 4, 5]
        # tf.cumsum(a) -> [0, 0, 0, 1, 1, 2]
        # tf.gather(count_to_substract, tf.cumsum(a)) -> [0, 0, 0, 3, 3, 5]
        # new_word_indices -> [0, 1, 2, 3, 4, 5] - [0, 0, 0, 3, 3, 5] = [0, 1, 2, 0, 1, 0]
        # new_word_indices is the concatenation of range(word_len(sentence))
        # for all sentences in units

        n_total_word_elements = max_token_seq_len*torch.ones_like(token_seq_lengths, device=device).sum()
        word_indices_flat = (idxs[:, 0] * max_token_seq_len + new_word_indices).to(torch.int64)
        #x_mask = torch.sum(torch.nn.functional.one_hot(word_indices_flat, n_total_word_elements), 0)
        #x_mask = x_mask.to(torch.bool)
        x_mask = torch.zeros(n_total_word_elements, dtype=torch.bool, device=device)
        x_mask[word_indices_flat] = torch.ones_like(word_indices_flat, device=device, dtype=torch.bool)
        # to get absolute indices we add max_token_seq_len:
        # idxs[:, 0] * max_token_seq_len -> [0, 0, 0, 1, 1, 2] * 2 = [0, 0, 0, 3, 3, 6]
        # word_indices_flat -> [0, 0, 0, 3, 3, 6] + [0, 1, 2, 0, 1, 0] = [0, 1, 2, 3, 4, 6]
        # total number of words in the batch (including paddings)
        # batch_size * max_token_seq_len -> 3 * 3 = 9
        # tf.one_hot(...) ->
        # [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
        #  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
        #  [0. 0. 0. 1. 0. 0. 0. 0. 0.]
        #  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
        #  [0. 0. 0. 0. 0. 0. 1. 0. 0.]]
        #  x_mask -> [1, 1, 1, 1, 1, 0, 1, 0, 0]

        nonword_indices_flat = (~x_mask).nonzero().squeeze(-1)
        # # y_idxs -> [5, 7, 8]

        # get a sequence of units corresponding to the start subtokens of the words
        # size: [n_words, n_features]
        
        elements = units[mask.bool()]

        # prepare zeros for paddings
        # size: [batch_size * TOKEN_seq_length - n_words, n_features]
        paddings = torch.zeros_like(nonword_indices_flat, dtype=elements.dtype).unsqueeze(-1).repeat(1,nf_int).to(device)

        # tensor_flat -> [x, x, x, x, x, 0, x, 0, 0]
        tensor_flat_unordered = torch.cat([elements, paddings])
        _, order_idx = torch.sort(torch.cat([word_indices_flat, nonword_indices_flat]))
        tensor_flat = tensor_flat_unordered[order_idx]

        tensor = torch.reshape(tensor_flat, (-1, max_token_seq_len, nf_int))
        # tensor -> [[x, x, x],
        #            [x, x, 0],
        #            [x, 0, 0]]

        return tensor