import os
import pickle
import torch.utils.data
import collections
import trw.train
import numpy as np


def write_pickle_simple(file, case):
    """
    Simply write each case (feature_name, feature_value)
    """
    assert isinstance(case, collections.Mapping), '`case` must be a dict-like'
    pickle.dump(case, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle_simple_one(file):
    """
    Read a single sample from a chunk
    :param file:
    :return:
    """
    return pickle.load(file)


def chunk_name(root, base_name, chunk_id):
    """
    Name of the data chunk
    Args:
        root: the folder where the chunk is stored
        base_name: the chunk base name
        chunk_id: the id of the chunk

    Returns:
        the path to the chunk
    """
    path = os.path.join(root, base_name + '-chunk{}'.format(chunk_id) + '.pkl')
    return path


def chunk_samples(root, base_name, samples, nb_samples_per_chunk=50, write_fn=write_pickle_simple, header_fn=None):
    """
    Split the cases in batches that can be loaded quickly, additionally, create
    a header containing the file position for each case in a chunk to enable random access
    of the cases within a chunk

    The header is extremely small, so it can be loaded in memory for very large datasets

    :param root: the root directory where the chunked cases will be exported
    :param base_name: the base name of each chunk
    :param samples: the cases. Must be a list of dictionary of (feature_name, feature_value)
    :param nb_samples_per_chunk: the maximum number of cases per chunk
    :param write_fn: defines how the cases are serialized
    :param header_fn: a function `f(sample)->dict` that will be used to populate a header
    :return: the number of chunks
    """
    file = None
    chunk_id = 0
    tellp = []
    headers = []
    for case_id, c in enumerate(samples):
        # create a new chunk
        if case_id % nb_samples_per_chunk == 0:
            tellp = []
            chunk_id += 1
            if file is not None:
                file.close()  # clear the previous file
            else:
                chunk_id = 0

            file = open(chunk_name(root, base_name, chunk_id), 'wb')

        tellp.append(file.tell())
        if header_fn is not None:
            header = header_fn(c)
            headers.append(header)
        write_fn(file, c)

        if case_id > 0 and case_id % nb_samples_per_chunk == (nb_samples_per_chunk - 1):
            with open(os.path.join(root, base_name + '-chunk{}_header'.format(chunk_id) + '.pkl'), 'wb') as file_header:
                pickle.dump(tellp, file_header)
                pickle.dump(headers, file_header)

    if file is not None:
        file.close()

    return chunk_id + 1


def read_whole_chunk(chunk_path, read_fn=read_pickle_simple_one):
    """
    Read the whole chunk at once
    """
    cases = []
    with open(chunk_path, 'rb') as f:
        try:
            while True:
                c = read_fn(f)
                cases.append(c)
        except EOFError:
            pass
    return cases


def _read_whole_chunk_sequence(batch):
    assert isinstance(batch, collections.Mapping), 'must be a mapping!'
    chunk_path = batch['chunks_path']
    #print('Reading chunk=', chunk_path)
    assert len(chunk_path) == 1, 'support only one chunk loading at a time!'
    cases = read_whole_chunk(chunk_path[0])

    # reformat the case to have `feature_name, all_sample_feature_values` format
    return torch.utils.data.dataloader.default_collate(cases)


def create_chunk_sequence(root, base_name, nb_chunks, chunk_start=0, nb_workers=0, max_jobs_at_once=None, sampler=trw.train.SamplerRandom(batch_size=1)):
    """
    Create an asynchronously loaded sequence of chunks

    Args:
        root: the directory where the chnuks are stored
        base_name: the basename of the chnuks
        nb_chunks: the number of chunks to load
        chunk_start: the starting chunk
        nb_workers: the number of workers dedicated to load the chunks
        max_jobs_at_once: the maximum number of jobs allocated at once
        sampler: the sampler of the chunks to be loaded

    Returns:
        a sequence
    """
    chunks_path = [chunk_name(root, base_name, chunk_id) for chunk_id in range(chunk_start, chunk_start + nb_chunks)]
    chunks_path = np.asarray(chunks_path)
    sequence = trw.train.SequenceArray({'chunks_path': chunks_path}, sampler=sampler).map(_read_whole_chunk_sequence, nb_workers=nb_workers, max_jobs_at_once=max_jobs_at_once)
    return sequence


def create_chunk_reservoir(
        root,
        base_name,
        nb_chunks,
        max_reservoir_samples,
        min_reservoir_samples,
        chunk_start=0,
        nb_workers=1,
        input_sampler=trw.train.SamplerRandom(batch_size=1),
        reservoir_sampler=None,
        maximum_number_of_samples_per_epoch=None,
        max_jobs_at_once=None):
    """
    Create a reservoir of chunk asynchronously loaded

    Args:
        root: the directory where the chnuks are stored
        base_name: the basename of the chnuks
        nb_chunks: the number of chunks to load
        max_reservoir_samples: the size of the reservoir
        min_reservoir_samples: the minimum of samples to be loaded before starting a sequence
        chunk_start: the starting chunk
        nb_workers: the number of workers dedicated to load the chunks
        input_sampler: the sampler of the chunks to be loaded
        reservoir_sampler: the sampler used for the reservoir
        maximum_number_of_samples_per_epoch: the maximum number of samples generated before the sequence is interrupted
        max_jobs_at_once: maximum number of jobs in the input queue

    Returns:
        a sequence
    """
    chunks_path = [chunk_name(root, base_name, chunk_id) for chunk_id in range(chunk_start, chunk_start + nb_chunks)]
    chunks_path = np.asarray(chunks_path)
    sequence = trw.train.SequenceArray({'chunks_path': chunks_path}, sampler=input_sampler).\
        async_reservoir(
            max_reservoir_samples=max_reservoir_samples,
            min_reservoir_samples=min_reservoir_samples,
            function_to_run=_read_whole_chunk_sequence,
            nb_workers=nb_workers,
            reservoir_sampler=reservoir_sampler,
            maximum_number_of_samples_per_epoch=maximum_number_of_samples_per_epoch,
            max_jobs_at_once=max_jobs_at_once,
        )
    return sequence


class DatasetChunked(torch.utils.data.Dataset):
    """
    Chunked datasets to enable larger than RAM datasets to be processed

    The idea is to have a very large dataset split in chunks. Each chunks contains `N` samples. Chunks are loaded in two parts:
    * the sample data: a binary file containing `N` samples. The samples are only loaded when requested
    * the header: this is loaded when the dataset is instantiated, it contains header description (e.g., file offset position per sample)
      and custom attributes

    Each sample within a chunk can be independently loaded
    """
    def __init__(self, root, base_name, chunk_id, reader_one_fn=read_pickle_simple_one):
        super().__init__()
        self.root = root
        self.base_name = base_name
        self.chunk_id = chunk_id
        self.reader_one_fn = reader_one_fn

        header_path = os.path.join(root, base_name + '-chunk{}_header'.format(chunk_id) + '.pkl')
        with open(header_path, 'rb') as f:
            self.sample_file_position = pickle.load(f)
            self.headers = pickle.load(f)

    def __len__(self):
        return len(self.sample_file_position)

    def __getitem__(self, item):
        path = os.path.join(self.root, self.base_name + '-chunk{}'.format(self.chunk_id) + '.pkl')
        with open(path, 'rb') as f:
            f.seek(self.sample_file_position[item])
            c = self.reader_one_fn(f)
        return c
