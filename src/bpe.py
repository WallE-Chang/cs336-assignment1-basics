#!/usr/bin/env python3
# coding=utf-8
"""
Author: changwanli
since: 2025-06-09 16:58:05
LastTime: 2025-06-11 18:04:42
LastAuthor: changwanli
message: 
Copyright (c) 2023 Wuhan Artificial Intelligence Research. All Rights Reserved 
"""
import os
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO, Dict, List, NamedTuple, Set, Tuple

import regex as re


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



GPT2_REGEX_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(text, regex_pattern=GPT2_REGEX_PATTERN, special_tokens: List[str]=[]) -> List[str]:
    # split by special tokens
    splitted_text_list = re.split(r'|'.join(map(re.escape, special_tokens)), text)
    tokens = []
    for splitted_text in splitted_text_list:
        if regex_pattern is None:
            tokens.extend(splitted_text.strip('\n').split())
        else:
            tokens.extend(re.findall(regex_pattern, splitted_text))
    return tokens
        




def pre_tokenize_from_file(file_path:str, start: int = 0, end: int = -1, regex_pattern=GPT2_REGEX_PATTERN, special_tokens: List[str]=[]) -> List[str]:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pre_tokenize(chunk, regex_pattern, special_tokens=special_tokens)
        

def pre_tokenize_from_file_parallel(file_path: str, num_processes: int, regex_pattern=GPT2_REGEX_PATTERN, special_tokens: List[str]=[]) -> List[str]:
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes,split_special_token="<|endoftext|>".encode("utf-8"))
    with Pool(num_processes) as pool:
        results = pool.starmap(pre_tokenize_from_file, [(file_path, boundaries[i], boundaries[i + 1], regex_pattern,special_tokens  ) for i in range(len(boundaries) - 1)])
    return [token for sublist in results for token in sublist]  # Flatten the list of lists
    




def visualize_pair(pair: Tuple[Tuple[int], Tuple[int]]):
    return ' '.join([bytes(b).decode('utf-8', errors='ignore') for b in pair])


def get_frequency_table(pre_tokenized_corpus: List[str]) -> Dict[Tuple[Tuple[int], ...], int]:
    # count the frequency of each token
    get_frequency_table = defaultdict(int)  # {vacab: frequency}
    for pre_token in pre_tokenized_corpus:
        get_frequency_table[tuple([(item,) for item in pre_token.encode('utf8')])] += 1

    return get_frequency_table


def visualize_frequency_table(frequency_table: Dict[Tuple[Tuple[int], ...], int]) -> str:
    """
    Visualize the frequency table as a string.

    Args:
        frequency_table (Dict[Tuple[Tuple[int], ...], int]): The frequency table.

    Returns:
        str: A string representation of the frequency table.
    """
    lines = []
    for token_tuple, count in frequency_table.items():
        token = [(bytes(b).decode('utf-8', errors='ignore')) for b in token_tuple]
        lines.append(f'{token}: {count}')
    return '\n'.join(lines)


def get_pari_2_counts(frequency_table: Dict[Tuple[Tuple[int], ...], int]):
    pair_2_counts = {}
    for token_tuple, count in frequency_table.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_2_counts[pair] = pair_2_counts.get(pair, 0) + count
    return pair_2_counts

def get_pari_2_counts_from_tokens(tokens: List[Tuple[int, ...]], count:int):
    pair_2_counts = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pair_2_counts[pair] = pair_2_counts.get(pair, 0) + count
    return pair_2_counts

def find_max_frequent_pair(pair_2_counts: Dict[Tuple[Tuple[int, ...],
                                                     Tuple[int, ...]],
                                               int]) -> Tuple[Tuple[Tuple[int, ...],
                                                                    Tuple[int, ...]],
                                                              int]:
    max_frequent_pair_count = max(pair_2_counts.values())
    max_frequent_pair_candidates = [
        pair for pair, count in pair_2_counts.items() if count == max_frequent_pair_count]
    # max_frequent_pair_candidates = list(map(lambda x: bytes(((*x[0], *x[1]))), max_frequent_pair_candidates))  # Convert to bytes
    max_frequent_pair = max(max_frequent_pair_candidates)
    return max_frequent_pair, max_frequent_pair_count


def merge_pair(frequency_table: Dict[Tuple[Tuple[int], ...], int], pair: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Dict[Tuple[Tuple[int], ...], int]:
    """
    Merge a pair of tokens in the frequency table.

    Args:
        frequency_table (Dict[Tuple[Tuple[int], ...], int]): The frequency table.
        pair (Tuple[Tuple[int, ...], Tuple[int, ...]]): The pair of tokens to merge.

    Returns:
        Dict[Tuple[Tuple[int], ...], int]: The updated frequency table after merging the pair.
    """
    pair_flattened = (*pair[0], *pair[1])  # Create a new token by merging the pair
    new_frequency_table = {}
    for token_tuple, count in frequency_table.items():
        i = 0
        new_token_tuple = []
        while i < len(token_tuple):
            if token_tuple[i] == pair[0] and i + 1 < len(token_tuple) and token_tuple[i + 1] == pair[1]:
                new_token_tuple.append(pair_flattened)
                i += 2  # Skip the next token as it is merged
            else:
                new_token_tuple.append(token_tuple[i])
                i += 1
        new_token_tuple = tuple(new_token_tuple)
        new_frequency_table[new_token_tuple] = count
    return new_frequency_table

def print_func_time(func):
    from line_profiler import LineProfiler
    def wrapper(*args, **kwargs):  # 指定宇宙无敌参数
        lp = LineProfiler()
        lp_wrapper = lp(func)
        result=lp_wrapper(*args, **kwargs)
        lp.print_stats()
        return result
    return wrapper  # 返


class BpeTokenizer():

    
    @staticmethod
    # @print_func_time
    def train(input_path: str, vocab_size: int, special_tokens: List[str],
              num_workers=1, verbose=False) -> Tuple[Dict[int, bytes],
                                      List[Tuple[bytes, bytes]]]:
        """
        Train a BPE tokenizer.

        Args:
            input_path (str): path to a text file with BPE tokenizer training data.
            vocab_size (int): A non-negative integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
            special_tokens (List[str]): A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.
            num_workers (int): Number of worker processes to use for pre-tokenization. If -1, uses all available CPU cores.

        Returns:
            Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]: 
                vocab (Dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
                merges (List[Tuple[bytes, bytes]]): A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
        """
        special_tokens = list(dict.fromkeys(
            [token for token in special_tokens], None))  # Remove duplicates while preserving order

        max_merges = vocab_size - len(special_tokens) - 256  # 256 is the size of the initial byte vocabulary


        # pre-tokenize the corpus
        pre_tokenized_corpus = pre_tokenize_from_file_parallel(input_path, num_processes=4, special_tokens=special_tokens)
        frequency_table = get_frequency_table(pre_tokenized_corpus)
        # print(f'visualize frequency table:\n{visualize_frequency_table(frequency_table)}')

        merges = []
        for merge_count in range(max_merges):
            # print(f'Iteration {merge_count + 1}/{max_merges}')
            pair_2_counts = get_pari_2_counts(frequency_table)
            if not pair_2_counts:
                break
            max_frequent_pair, max_frequent_pair_count = find_max_frequent_pair(pair_2_counts)
            if verbose:
                print(f'Merge Count: {merge_count};Found max frequent pair: {visualize_pair(max_frequent_pair)}, count: {max_frequent_pair_count}')
            merges.append(tuple(map(bytes, max_frequent_pair)))  # Convert to bytes
            frequency_table = merge_pair(frequency_table, max_frequent_pair)
            # print(f'visualize frequency table after merging:\n{visualize_frequency_table(frequency_table)}')
        print('done')

        vocab = {}
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode('utf8')
        for i in range(256):
            vocab[len(vocab)] = bytes([i])
        for merge in merges:
            vocab[len(vocab)] = merge[0] + merge[1]

        return vocab, merges


if __name__ == '__main__':
    # Example usage
    input_path = '/mnt/lustre/changwanli/project/2025/cs336/spring2024-assignment1-basics/tests/fixtures/corpus.en'
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    tokenizer = BpeTokenizer()
    from time import time
    start_time = time()
    vocab, merges = tokenizer.train(input_path, vocab_size, special_tokens, num_workers=1)
    end_time = time()
    print(f'Training time: {end_time - start_time:.2f} seconds')