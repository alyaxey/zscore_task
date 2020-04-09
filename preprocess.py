from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np


class ZScoreProcessor(object):
    def __init__(self):
        # accumulate features sum
        self.sum = 0
        # accumulate squared features sum
        self.sum2 = 0
        # accumulate rows count
        self.n = 0
        # calculate features mean at the end
        self.mean = None
        # calculate features unbiased standard deviation at the end
        self.std = None
        # calculate output column names at the end
        self.columns = None

    def preprocess(self, data: np.ndarray):
        self.sum += np.sum(data, axis=0)
        self.sum2 += np.sum(data ** 2, axis=0)
        self.n += data.shape[0]

    def process(self, kind: int, data: np.ndarray) -> pd.DataFrame:
        if self.mean is None:
            self.mean = self.sum / self.n
            mean2 = self.sum2 / self.n
            self.std = np.sqrt(self.n / (self.n - 1) * (mean2 - self.mean ** 2))
            self.columns = [f'feature_{kind}_stand_{i}' for i in range(data.shape[1])]
        # replace nan no zero in case of zero standard deviation
        result = np.nan_to_num((data - self.mean) / self.std)
        return pd.DataFrame(data=result, columns=self.columns)


class MaxAbsMeanDiffProcessor(object):
    def __init__(self):
        # accumulate features sum
        self.sum = 0
        # accumulate rows count
        self.n = 0
        # calculate features mean at the end
        self.mean = None

    def preprocess(self, data: np.ndarray):
        self.sum += np.sum(data, axis=0)
        self.n += data.shape[0]

    def process(self, kind: int, data: np.ndarray) -> pd.DataFrame:
        if self.mean is None:
            self.mean = self.sum / self.n
        max_index = np.argmax(data, axis=1)
        max_abs_mean_diff = np.abs(np.max(data, axis=1) - self.mean[max_index])
        return pd.DataFrame({
            f'max_feature_{kind}_index': max_index,
            f'max_feature_{kind}_abs_mean_diff': max_abs_mean_diff})


def preprocess(mappers: Dict[int, List], in_path: Path, out_path: Path, chunksize: int = 1000):
    # first iteration - preprocess, second - write output
    for iteration in range(2):
        # read tsv file in chunks to limit memory usage
        for i, df in enumerate(pd.read_table(in_path, chunksize=chunksize)):
            # transform features from Series of str to numpy array
            features = df.features.str.split(',', expand=True).astype(float).values
            # for now we're working with one kind only because not sure how it would be present in input
            kind = int(features[0, 0])
            features = features[:, 1:]
            # prepare output by parts
            outputs = [df[['id_job']].reset_index(drop=True)]
            for p in mappers[kind]:
                if iteration == 0:
                    p.preprocess(features)
                else:
                    outputs.append(p.process(kind, features))
            if iteration == 1:
                # truncate file and write header during first write only, then append without header
                pd.concat(outputs, axis=1)\
                    .to_csv(out_path, sep='\t', index=False, header=i == 0, mode='w' if i == 0 else 'a')


if __name__ == '__main__':
    """
    Would be nice to use Apache Spark in practice, but I assume this task is about manual implementation because all
    data is in one big file and other libraries are not mentioned. Also not sure how different feature types could be
    present in future, in one line or different. So partly ignoring this part. Further work - extract common code like
    mean calculation.
    """
    # list of processors for each feature kind
    mappers = {2: [ZScoreProcessor(), MaxAbsMeanDiffProcessor()]}
    preprocess(mappers, in_path=Path('data/test.tsv'), out_path=Path('data/test_proc.tsv'))
