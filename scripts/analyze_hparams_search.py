description = """The purpose of this script is to analyze the hyper-parameter search and try
to make sense of the hyper-parameters by plotting the importance hyper-parameter,
loss variation relative to hyper-parameter as well as hyper-parameter covariance"""

import trw
import argparse
import os


parser = argparse.ArgumentParser(description=description)
parser.add_argument('--hyper_parameter_root', help='the root directory where the hyper-parameter results are stored')
parser.add_argument('--output_path', help='path where to export the analysis')
parser.add_argument('--pattern', help='the pattern of the hyper-parameter result files', default='hparams-*.pkl')
args = parser.parse_args()

pattern = os.path.join(args.hyper_parameter_root, args.pattern)
trw.hparams.analyse_hyperparameters(pattern, args.output_path)
