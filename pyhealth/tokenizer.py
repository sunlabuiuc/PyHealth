from typing import List, Optional, Tuple


class Vocabulary:
    """Vocabulary class for mapping between tokens and indices."""

    def __init__(self, tokens: List[str], special_tokens: Optional[List[str]] = None):
        """Initializes the vocabulary.

        This function initializes the vocabulary by adding the special tokens first
        and then the tokens. The order of the tokens is preserved.

        If <unk> is not provided in the special_tokens, then the tokenizer
        will raise an exception if an unknown token is encountered.

        If padding is performed on the input tokens, padding token <pad> should always
        be added to the special_tokens.

        Args:
            tokens: List[str], list of tokens in the vocabulary.
            special_tokens: Optional[List[str]], list of special tokens to add to
                the vocabulary. (e.g., <pad>, <unk>). Default is empty list.

        Note:
            If vocabulary is used to convert output labels to indices, one should
                be very careful about the special tokens.
        """
        if special_tokens is None:
            special_tokens = []
        all_tokens = special_tokens + tokens
        self.token2idx = {}
        self.idx2token = {}
        self.idx = 0
        for token in all_tokens:
            self.add_token(token)

    def add_token(self, token):
        """Adds a token to the vocabulary."""
        if token not in self.token2idx:
            self.token2idx[token] = self.idx
            self.idx2token[self.idx] = token
            self.idx += 1

    def __call__(self, token):
        """Retrieves the index of the token.

        Note that if the token is not in the vocabulary, this function will try to
        return the index of <unk>. If <unk> is not in the vocabulary,
        an exception will be raised.
        """
        if token not in self.token2idx:
            if "<unk>" in self.token2idx:
                return self.token2idx["<unk>"]
            else:
                raise ValueError("Unknown token: {}".format(token))
        return self.token2idx[token]

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.token2idx)

    def __contains__(self, token):
        return token in self.token2idx


class Tokenizer:
    """Tokenizer class for converting tokens to indices and vice versa.

    This class will build a vocabulary from the provided tokens and provide the
    functionality to convert tokens to indices and vice versa. This class also
    provides the functionality to tokenize a batch of data.
    
    Examples:
            >>> from pyhealth.tokenizer import Tokenizer
            >>> token_space = ['A01A', 'A02A', 'A02B', 'A02X', 'A03A', 'A03B', 'A03C', 'A03D', 'A03E', \
            ...                'A03F', 'A04A', 'A05A', 'A05B', 'A05C', 'A06A', 'A07A', 'A07B', 'A07C', \
            ...                'A07D', 'A07E', 'A07F', 'A07X', 'A08A', 'A09A', 'A10A', 'A10B', 'A10X', \
            ...                'A11A', 'A11B', 'A11C', 'A11D', 'A11E', 'A11G', 'A11H', 'A11J', 'A12A', \
            ...                'A12B', 'A12C', 'A13A', 'A14A', 'A14B', 'A16A']
            >>> tokenizer = Tokenizer(tokens=token_space, special_tokens=["<pad>", "<unk>"])
    """

    def __init__(self, tokens: List[str], special_tokens: Optional[List[str]] = None):
        """Initializes the tokenizer.

        Args:
            tokens: List[str], list of tokens in the vocabulary.
            special_tokens: Optional[List[str]], list of special tokens to add to
                the vocabulary. (e.g., <pad>, <unk>). Default is empty list.
        """
        self.vocabulary = Vocabulary(tokens=tokens, special_tokens=special_tokens)

    def get_padding_index(self):
        """Returns the index of the padding token."""
        return self.vocabulary("<pad>")

    def get_vocabulary_size(self):
        """Returns the size of the vocabulary.

        Examples:
            >>> tokenizer.get_vocabulary_size()
            44
        """
        return len(self.vocabulary)

    def convert_tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """Converts a list of tokens to indices.
        
        Examples:
            >>> tokens = ['A03C', 'A03D', 'A03E', 'A03F', 'A04A', 'A05A', 'A05B', 'B035', 'C129']
            >>> indices = tokenizer.convert_tokens_to_indices(tokens)
            >>> print(indices)
            [8, 9, 10, 11, 12, 13, 14, 1, 1]
        """
        return [self.vocabulary(token) for token in tokens]

    def convert_indices_to_tokens(self, indices: List[int]) -> List[str]:
        """Converts a list of indices to tokens.
        
        Examples:
            >>> indices = [0, 1, 2, 3, 4, 5]
            >>> tokens = tokenizer.convert_indices_to_tokens(indices)
            >>> print(tokens)
            ['<pad>', '<unk>', 'A01A', 'A02A', 'A02B', 'A02X']
        """
        return [self.vocabulary.idx2token[idx] for idx in indices]

    def batch_encode_2d(
        self,
        batch: List[List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
    ):
        """Converts a list of lists of tokens (2D) to indices.

        Args:
            batch: List of lists of tokens to convert to indices.
            padding: whether to pad the tokens to the max number of tokens in
                the batch (smart padding).
            truncation: whether to truncate the tokens to max_length.
            max_length: maximum length of the tokens. This argument is ignored
                if truncation is False.
        
        Examples:
            >>> tokens = [
            ...     ['A03C', 'A03D', 'A03E', 'A03F'],
            ...     ['A04A', 'B035', 'C129']
            ... ]

            >>> indices = tokenizer.batch_encode_2d(tokens)
            >>> print ('case 1:', indices)
            case 1: [[8, 9, 10, 11], [12, 1, 1, 0]]

            >>> indices = tokenizer.batch_encode_2d(tokens, padding=False)
            >>> print ('case 2:', indices)
            case 2: [[8, 9, 10, 11], [12, 1, 1]]

            >>> indices = tokenizer.batch_encode_2d(tokens, max_length=3)
            >>> print ('case 3:', indices)
            case 3: [[9, 10, 11], [12, 1, 1]]
        """

        if truncation:
            batch = [tokens[-max_length:] for tokens in batch]
        if padding:
            batch_max_length = max([len(tokens) for tokens in batch])
            batch = [
                tokens + ["<pad>"] * (batch_max_length - len(tokens))
                for tokens in batch
            ]
        return [[self.vocabulary(token) for token in tokens] for tokens in batch]

    def batch_decode_2d(
        self,
        batch: List[List[int]],
        padding: bool = False,
    ):
        """Converts a list of lists of indices (2D) to tokens.

        Args:
            batch: List of lists of indices to convert to tokens.
            padding: whether to keep the padding tokens from the tokens.
        
        Examples:
            >>> indices = [
            ...     [8, 9, 10, 11],
            ...     [12, 1, 1, 0]
            ... ]

            >>> tokens = tokenizer.batch_decode_2d(indices)
            >>> print ('case 1:', tokens)
            case 1: [['A03C', 'A03D', 'A03E', 'A03F'], ['A04A', '<unk>', '<unk>']]

            >>> tokens = tokenizer.batch_decode_2d(indices, padding=True)
            >>> print ('case 2:', tokens)
            case 2: [['A03C', 'A03D', 'A03E', 'A03F'], ['A04A', '<unk>', '<unk>', '<pad>']]
        """

        batch = [[self.vocabulary.idx2token[idx] for idx in tokens] for tokens in batch]
        if not padding:
            return [[token for token in tokens if token != "<pad>"] for tokens in batch]
        return batch

    def batch_encode_3d(
        self,
        batch: List[List[List[str]]],
        padding: Tuple[bool, bool] = (True, True),
        truncation: Tuple[bool, bool] = (True, True),
        max_length: Tuple[int, int] = (10, 512),
    ):
        """Converts a list of lists of lists of tokens (3D) to indices.

        Args:
            batch: List of lists of lists of tokens to convert to indices.
            padding: a tuple of two booleans indicating whether to pad the tokens
                to the max number of tokens and visits (smart padding).
            truncation: a tuple of two booleans indicating whether to truncate the
                tokens to the corresponding element in max_length
            max_length: a tuple of two integers indicating the maximum length of the
                tokens along the first and second dimension. This argument is ignored
                if truncation is False.
        
        Examples:
                >>> tokens = [
                ...     [
                ...         ['A03C', 'A03D', 'A03E', 'A03F'],
                ...         ['A08A', 'A09A'],
                ...     ],
                ...     [
                ...         ['A04A', 'B035', 'C129'],
                ...     ]
                ... ]

                >>> indices = tokenizer.batch_encode_3d(tokens)
                >>> print ('case 1:', indices)
                case 1: [[[8, 9, 10, 11], [24, 25, 0, 0]], [[12, 1, 1, 0], [0, 0, 0, 0]]]

                >>> indices = tokenizer.batch_encode_3d(tokens, padding=(False, True))
                >>> print ('case 2:', indices)
                case 2: [[[8, 9, 10, 11], [24, 25, 0, 0]], [[12, 1, 1, 0]]]

                >>> indices = tokenizer.batch_encode_3d(tokens, padding=(True, False))
                >>> print ('case 3:', indices)
                case 3: [[[8, 9, 10, 11], [24, 25]], [[12, 1, 1], [0]]]

                >>> indices = tokenizer.batch_encode_3d(tokens, padding=(False, False))
                >>> print ('case 4:', indices)
                case 4: [[[8, 9, 10, 11], [24, 25]], [[12, 1, 1]]]

                >>> indices = tokenizer.batch_encode_3d(tokens, max_length=(2,2))
                >>> print ('case 5:', indices)
                case 5: [[[10, 11], [24, 25]], [[1, 1], [0, 0]]]
        """
        if truncation[0]:
            batch = [tokens[-max_length[0] :] for tokens in batch]
        if truncation[1]:
            batch = [
                [tokens[-max_length[1] :] for tokens in visits] for visits in batch
            ]
        if padding[0]:
            batch_max_length = max([len(tokens) for tokens in batch])
            batch = [
                tokens + [["<pad>"]] * (batch_max_length - len(tokens))
                for tokens in batch
            ]
        if padding[1]:
            batch_max_length = max(
                [max([len(tokens) for tokens in visits]) for visits in batch]
            )
            batch = [
                [
                    tokens + ["<pad>"] * (batch_max_length - len(tokens))
                    for tokens in visits
                ]
                for visits in batch
            ]
        return [
            [[self.vocabulary(token) for token in tokens] for tokens in visits]
            for visits in batch
        ]

    def batch_decode_3d(
        self,
        batch: List[List[List[int]]],
        padding: bool = False,
    ):
        """Converts a list of lists of lists of indices (3D) to tokens.

        Args:
            batch: List of lists of lists of indices to convert to tokens.
            padding: whether to keep the padding tokens from the tokens.
        
        Examples:
            >>> indices = [
            ...     [
            ...         [8, 9, 10, 11], 
            ...         [24, 25, 0, 0]
            ...     ], 
            ...     [
            ...         [12, 1, 1, 0], 
            ...         [0, 0, 0, 0]
            ...     ]
            ... ]

            >>> tokens = tokenizer.batch_decode_3d(indices)
            >>> print ('case 1:', tokens)
            case 1: [[['A03C', 'A03D', 'A03E', 'A03F'], ['A08A', 'A09A']], [['A04A', '<unk>', '<unk>']]]

            >>> tokens = tokenizer.batch_decode_3d(indices, padding=True)
            >>> print ('case 2:', tokens)
            case 2: [[['A03C', 'A03D', 'A03E', 'A03F'], ['A08A', 'A09A', '<pad>', '<pad>']], [['A04A', '<unk>', '<unk>', '<pad>'], ['<pad>', '<pad>', '<pad>', '<pad>']]]
        """
        batch = [
            self.batch_decode_2d(batch=visits, padding=padding) for visits in batch
        ]
        if not padding:
            batch = [[visit for visit in visits if visit != []] for visits in batch]
        return batch


if __name__ == "__main__":
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
    tokenizer = Tokenizer(tokens=tokens, special_tokens=["<pad>", "<unk>"])
    print(tokenizer.get_vocabulary_size())

    out = tokenizer.convert_tokens_to_indices(["a", "b", "c", "d", "e", "z"])
    print(out)
    print(tokenizer.convert_indices_to_tokens(out))

    out = tokenizer.batch_encode_2d(
        [["a", "b", "c", "e", "z"], ["a", "b", "c", "d", "e", "z"]],
        padding=True,
        truncation=True,
        max_length=10,
    )
    print(out)
    print(tokenizer.batch_decode_2d(out, padding=False))

    out = tokenizer.batch_encode_3d(
        [
            [["a", "b", "c", "e", "z"], ["a", "b", "c", "d", "e", "z"]],
            [["a", "b", "c", "e", "z"], ["a", "b", "c", "d", "e", "z"], ["c", "f"]],
        ],
        padding=(True, True),
        truncation=(True, True),
        max_length=(10, 10),
    )
    print(out)
    print(tokenizer.batch_decode_3d(out, padding=False))
