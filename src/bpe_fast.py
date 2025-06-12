#!/usr/bin/env python3
# coding=utf-8
"""
Author: changwanli
since: 2025-06-09 16:58:05
LastTime: 2025-06-12 17:51:38
LastAuthor: changwanli
message: 
Copyright (c) 2023 Wuhan Artificial Intelligence Research. All Rights Reserved 
"""
import os
from collections import Counter, defaultdict
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from typing import BinaryIO, Dict, List, NamedTuple, Set, Tuple

import regex as re
from ordered_set import OrderedSet


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


def pre_tokenize(text, regex_pattern=GPT2_REGEX_PATTERN, special_tokens: List[str] = []) -> List[bytes]:
    # split by special tokens
    splitted_text_list = re.split(r'|'.join(map(re.escape, special_tokens)), text)
    tokens = []
    for splitted_text in splitted_text_list:
        if regex_pattern is None:
            tokens.extend(splitted_text.strip('\n').split())
        else:
            tokens.extend(re.findall(regex_pattern, splitted_text))
    return [token.encode('utf-8') for token in tokens]


def pre_tokenize_from_file(file_path: str, start: int = 0, end: int = -1, regex_pattern=GPT2_REGEX_PATTERN,
                           special_tokens: List[str] = []) -> List[bytes]:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pre_tokenize(chunk, regex_pattern, special_tokens=special_tokens)


def pre_tokenize_from_file_parallel(file_path: str, num_processes: int, regex_pattern=GPT2_REGEX_PATTERN,
                                    special_tokens: List[str] = []) -> List[bytes]:
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token="<|endoftext|>".encode("utf-8"))
    with Pool(num_processes) as pool:
        results = pool.starmap(
            pre_tokenize_from_file,
            [(file_path, boundaries[i],
              boundaries[i + 1],
              regex_pattern, special_tokens) for i in range(len(boundaries) - 1)])
    return [token for sublist in results for token in sublist]  # Flatten the list of lists


def visualize_pair(pair: Tuple[Tuple[int], Tuple[int]]):
    return ' '.join([bytes(b).decode('utf-8', errors='ignore') for b in pair])


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


def get_pari_2_counts_from_tokens(tokens: List[Tuple[int, ...]], count: int):
    pair_2_counts = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pair_2_counts[pair] = pair_2_counts.get(pair, 0) + count
    return pair_2_counts


def find_max_frequent_pair(pair_2_counts: Dict[Tuple[bytes,
                                                     bytes],
                                               int]) -> Tuple[Tuple[bytes, bytes],
                                                              int]:
    max_frequent_pair_count = max(pair_2_counts.values())
    max_frequent_pair_candidates = [
        pair for pair, count in pair_2_counts.items() if count == max_frequent_pair_count]
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
        result = lp_wrapper(*args, **kwargs)
        lp.print_stats()
        return result
    return wrapper  # 返


# def is_sorted(iterable: List[int]) -> bool:
#     """
#     Check if a list is sorted in ascending order.

#     Args:
#         iterable (List[int]): The list to check.

#     Returns:
#         bool: True if the list is sorted, False otherwise.
#     """
#     return all(iterable[i] <= iterable[i + 1] for i in range(len(iterable) - 1))


#         pair_start_index_list = list(pair_start_index_set)
#         token_idx = 0
#         new_word = []
#         while token_idx < len(word):
#             if (not pair_start_index_list):
#                 new_word.append(word[token_idx])
#                 token_idx += 1
#             elif (token_idx < pair_start_index_list[0]):
#                 new_word.append(word[token_idx])
#                 token_idx += 1
#             elif (token_idx > pair_start_index_list[0]):
#                 pair_start_index_list.pop(0)
#             else:
#                 new_word.append(max_frequent_pair[0] + max_frequent_pair[1])
#                 token_idx += 2
#                 pair_start_index_list.pop(0)
#         if sanity_check:
#             assert len(pair_start_index_list) == 0, f'pair_start_index_list is not empty: {pair_start_index_list}'
#             assert b"".join(word) == b"".join(new_word), f'word: {word}, new_word: {new_word}'
#         byte_word_unique_list[word_idx] = new_word

def merge_word(word, pair_start_index_set, max_frequent_pair, sanity_check=False):

    pair_start_index_list = list(pair_start_index_set)
    token_idx = 0
    new_word = []
    while token_idx < len(word):
        if (not pair_start_index_list):
            new_word.append(word[token_idx])
            token_idx += 1
        elif (token_idx < pair_start_index_list[0]):
            new_word.append(word[token_idx])
            token_idx += 1
        elif (token_idx > pair_start_index_list[0]):
            pair_start_index_list.pop(0)
        else:
            new_word.append(max_frequent_pair[0] + max_frequent_pair[1])
            token_idx += 2
            pair_start_index_list.pop(0)
    if sanity_check:
        assert len(pair_start_index_list) == 0, f'pair_start_index_list is not empty: {pair_start_index_list}'
        assert b"".join(word) == b"".join(new_word), f'word: {word}, new_word: {new_word}'
    return new_word
    


def update(max_frequent_pair, pair_2_counts, byte_word_unique_list, byte_word_unique_frequency_list, pair_2_word_positions,sanity_check=False):
    # update  pair_2_counts, byte_word_unique_list, pair_2_word_positions
    affected_word_index_2_pair_start_index_set = deepcopy(pair_2_word_positions[max_frequent_pair])
    del pair_2_counts[max_frequent_pair]
    # sanity check
    for word_idx, pair_start_index_set in affected_word_index_2_pair_start_index_set.items():
        word = byte_word_unique_list[word_idx]
        if sanity_check:
            assert is_sorted(list(pair_start_index_set)), f'pair_start_index_set is not sorted: {pair_start_index_set}'
            for pair_index_in_word in pair_start_index_set:
                assert word[pair_index_in_word] == max_frequent_pair[0]
                assert word[pair_index_in_word+1] == max_frequent_pair[1]

    for word_idx, pair_start_index_set in affected_word_index_2_pair_start_index_set.items():
        word = byte_word_unique_list[word_idx]
        word_count = byte_word_unique_frequency_list[word_idx]

        # ----------------------------- 更新 pair_2_counts ----------------------------- #
        for pair_index_in_word in pair_start_index_set:
            # 处理旧 pair
            if pair_index_in_word > 0:
                left_pair = (word[pair_index_in_word-1], word[pair_index_in_word])
                pair_2_counts[left_pair] -= word_count
                if pair_2_counts[left_pair] == 0:
                    del pair_2_counts[left_pair]
            if pair_index_in_word + 2 < len(word):
                right_pair = (word[pair_index_in_word+1],  word[pair_index_in_word+2])
                pair_2_counts[right_pair] -= word_count
                if pair_2_counts[right_pair] == 0:
                    del pair_2_counts[right_pair]
            # 处理新 pair
            if pair_index_in_word > 0:
                new_left_pair = (word[pair_index_in_word-1], max_frequent_pair[0]+max_frequent_pair[1])
                pair_2_counts[new_left_pair] += word_count

            if pair_index_in_word + 2 < len(word):
                new_right_pair = (max_frequent_pair[0]+max_frequent_pair[1],  word[pair_index_in_word+2])
                pair_2_counts[new_right_pair] += word_count

        # ---------------------------------- 更新 byte_word_unique_list --------------------------------- #

        pair_start_index_list = list(pair_start_index_set)
        token_idx = 0
        new_word = []
        while token_idx < len(word):
            if (not pair_start_index_list):
                new_word.append(word[token_idx])
                token_idx += 1
            elif (token_idx < pair_start_index_list[0]):
                new_word.append(word[token_idx])
                token_idx += 1
            elif (token_idx > pair_start_index_list[0]):
                pair_start_index_list.pop(0)
            else:
                new_word.append(max_frequent_pair[0] + max_frequent_pair[1])
                token_idx += 2
                pair_start_index_list.pop(0)
        if sanity_check:
            assert len(pair_start_index_list) == 0, f'pair_start_index_list is not empty: {pair_start_index_list}'
            assert b"".join(word) == b"".join(new_word), f'word: {word}, new_word: {new_word}'
        byte_word_unique_list[word_idx] = new_word

        # ------------------------- 更新 pair_2_word_positions ------------------------- #
        # 删除旧的
        old_pair_set = set()
        for left_idx in range(len(word)-1):
            old_pair_set.add((word[left_idx], word[left_idx + 1]))

        for old_pair in old_pair_set:
            del pair_2_word_positions[old_pair][word_idx]

        # 添加新的
        for left_idx in range(len(new_word)-1):
            right_idx = left_idx + 1
            pair = (new_word[left_idx], new_word[right_idx])
            if word_idx in pair_2_word_positions[pair]:
                pair_2_word_positions[pair][word_idx].add(left_idx)
            else:
                pair_2_word_positions[pair][word_idx] = OrderedSet([left_idx])

            if sanity_check:
                assert is_sorted(list(pair_2_word_positions[pair][word_idx]))

    return pair_2_counts, byte_word_unique_list, pair_2_word_positions


class BpeTokenizer():

    @staticmethod
    # @print_func_time
    def train(input_path: str, vocab_size: int, special_tokens: List[str],
              num_workers=1, verbose=False, regex_pattern=GPT2_REGEX_PATTERN, sanity_check=False) -> Tuple[Dict[int, bytes],
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
        byte_words = pre_tokenize_from_file_parallel(
            input_path, num_processes=4, special_tokens=special_tokens, regex_pattern=regex_pattern)
        byte_words_frequency = Counter(byte_words)

        byte_word_unique_list, byte_word_unique_frequency_list = zip(*(byte_words_frequency.items()))
        byte_word_unique_list = [tuple(bytes([b]) for b in byte_word) for byte_word in byte_word_unique_list]

        del byte_words_frequency

        pair_2_counts = defaultdict(int)
        pair_2_word_positions = defaultdict(dict)  # {'word_iddex': OrderedSet[start_index1, start_index2, ...]}

        for byte_word_idx, byte_word in enumerate(byte_word_unique_list):
            count = byte_word_unique_frequency_list[byte_word_idx]
            for start_index in range(len(byte_word) - 1):
                pair = (byte_word[start_index], byte_word[start_index + 1])
                pair_2_counts[pair] += count
                if byte_word_idx in pair_2_word_positions[pair]:
                    pair_2_word_positions[pair][byte_word_idx].add(start_index)
                else:
                    pair_2_word_positions[pair][byte_word_idx] = OrderedSet([start_index])

        merges = []
        for merge_count in range(max_merges):
            # print(f'Iteration {merge_count + 1}/{max_merges}')
            if not pair_2_counts:
                break
            max_frequent_pair, max_frequent_pair_count = find_max_frequent_pair(pair_2_counts)
            if verbose:
                print(
                    f'Merge Count: {merge_count};Found max frequent pair: {max_frequent_pair}, count: {max_frequent_pair_count}')
            merges.append(max_frequent_pair)  # Convert to bytes
            pair_2_counts, byte_word_unique_list, pair_2_word_positions = update(
                max_frequent_pair, pair_2_counts, byte_word_unique_list, byte_word_unique_frequency_list, pair_2_word_positions, sanity_check=sanity_check)
            if verbose:
                print(pair_2_counts)
                print(byte_word_unique_list)
                print('========')

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
    # input_path = '/mnt/lustre/changwanli/project/2025/cs336/spring2024-assignment1-basics/src/test.txt'
    input_path = '/mnt/data/changwanli/project/my_project/2025/cs336_2025/cs336-assignment1-basics/tests/fixtures/corpus.en'
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    tokenizer = BpeTokenizer()
    from time import time
    start_time = time()
    vocab, merges = tokenizer.train(input_path, vocab_size, special_tokens, num_workers=1, regex_pattern=None)
    end_time = time()
    print(f'Training time: {end_time - start_time:.2f} seconds')
