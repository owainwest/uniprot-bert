# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf
import statistics

from tqdm import tqdm

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, 
    "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_integer(
    "k", 1, 
    "Size of kmer")

flags.DEFINE_integer(
    "gapfactor", 0, 
    "gapfactor=n gives an unsupported central gap of length 2n+1")

flags.DEFINE_float("masked_lm_prob", 0.15, 
    "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the maximum length.")

flags.DEFINE_bool(
    "do_hydro", False,
    "Whether or not to use local hydrophobicity predictions in training.")

flags.DEFINE_bool(
    "do_charge", False,
    "Whether or not to use local charge predictions in training.")

flags.DEFINE_bool(
    "do_pks", False,
    "Whether or not to use local predictions of pKa NH2, pKa COOH in training.")

flags.DEFINE_bool(
    "do_solubility", False,
    "Whether or not to use local predictions of solubility in training.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""
  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, 
    hydrophobicities=None, charges=None, pks=None, solubilities=None):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels
    self.hydrophobicities = hydrophobicities
    self.charges = charges
    self.pks = pks
    self.solubilities = solubilities

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    if self.hydrophobicities is not None:
        s += "hydrophobicities: %s\n" % (" ".join(
            [str(x) for x in self.hydrophobicities]))
    if self.charges is not None:
        s += "charges: %s\n" % (" ".join(
            [str(x) for x in self.charges]))
    if self.pks is not None:
        s += "pks: %s\n" % (" ".join(
            [str(x) for x in self.pks]))
    if self.solubilities is not None:
        s += "solubilities: %s\n" % (" ".join(
            [str(x) for x in self.solubilities]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)
    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)


    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

    if FLAGS.do_hydro:
        hydrophobicities = list(instance.hydrophobicities)
        hydrophobicity_weights = [1.0] * len(masked_lm_ids)
        while len(hydrophobicities) < max_predictions_per_seq:
            hydrophobicities.append(0)
            hydrophobicity_weights.append(0.0)
        features["hydrophobicities"] = create_int_feature(hydrophobicities)
        features["hydrophobicity_weights"] = create_float_feature(hydrophobicity_weights)

    if FLAGS.do_charge:
        charges = list(instance.charges)
        charge_weights = [1.0] * len(masked_lm_ids)
        while len(charges) < max_predictions_per_seq:
            charges.append(0)
            charge_weights.append(0.0)
        features["charges"] = create_int_feature(charges)
        features["charge_weights"] = create_float_feature(charge_weights)

    if FLAGS.do_pkss:
        pks = list(instance.pks)
        pk_weights = [1.0] * len(masked_lm_ids)
        while len(pks) < max_predictions_per_seq:
            pks.append(0)
            pk_weights.append(0.0)
        features["pks"] = create_int_feature(pks)
        features["pk_weights"] = create_float_feature(pk_weights)

    if FLAGS.do_solubility:
        solubilities = list(instance.solubilities)
        solubility_weights = [1.0] * len(masked_lm_ids)
        while len(solubilities) < max_predictions_per_seq:
            solubilities.append(0)
            solubility_weights.append(0.0)
        features["solubilities"] = create_int_feature(solubilities)
        features["solubility_weights"] = create_float_feature(solubility_weights)


    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.compat.v1.logging.info("*** Example ***")
      tf.compat.v1.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.compat.v1.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.compat.v1.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng, k, gapfactor):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    with tf.io.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for _ in tqdm(range(dupe_factor)):
    for document_index in range(len(all_documents)):
      instances.extend(
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng, k, gapfactor))

  rng.shuffle(instances)
  return instances


# k is size of each kmer
# gapfactor is: if you want a unsupported central gap of length 2n + 1, gapfactor is n
# eg - 3mers and 5-masking, gap factor is 0 (since unsupported center is size 1)
# 3 mers and 7-masking - gap factor is 1 for an unsupported 3-center
def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng, k=1, gapfactor=0):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP]
  max_num_tokens = max_seq_length - 2

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  i = 0
  while i < len(document):
    if len(document[i]) == 0:
      print('> Doc[i] was empty, i = ', i)
      continue

    lost = len(document[i]) - max_num_tokens
    tokens_a = document[i][:max_num_tokens]

    if (len(tokens_a) == 0):
      print('index', i)
      print(document[i])
      i += 1
      continue

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    (tokens, masked_lm_positions, masked_lm_labels, 
    hydrophobicities, charges, pks, solubilities) = create_masked_lm_predictions(
          tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, k, gapfactor)

    # Add feature to TrainingInstance for either sequence or token level features
    instance = TrainingInstance(
        tokens=tokens,
        segment_ids=segment_ids,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels,
        hydrophobicities=hydrophobicities,
        charges=charges,
        pks=pks,
        solubilities=solubilities)
    instances.append(instance)

    if lost > 10:
      document[i] = document[i][max_num_tokens:]
      continue

    i += 1

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", 
  ["index", "label", "hydrophobicity", "charge", "pks", "solubility"])


def get_hydrophobicity(peptide):
    acid_to_hydro = {
        'a': 41,
        'r': -14,
        'n': -28,
        'd': -55,
        'c': 49,
        'e': -31,
        'q': -10,
        'g': 0,
        'h': 8,
        'i': 99,
        'l': 97,
        'k': -23,
        'm': 74,
        'f': 100,
        'p': -46,
        's': -5,
        't': 13,
        'w': 97,
        'y': 63,
        'v': 76
    }
    DEFAULT_GUESS = statistics.median(acid_to_hydro.values())
    res = []
    for amino_acid in peptide:
        if amino_acid in acid_to_hydro:
            res.append(acid_to_hydro[amino_acid])
        else:
            res.append(DEFAULT_GUESS)
    return sum(res)

def get_charge(peptide):
    acid_to_charge = {
        'a': 0,
        'r': 1,
        'n': 0,
        'd': -1,
        'c': 0,
        'e': -1,
        'q': 0,
        'g': 0,
        'h': 1,
        'i': 0,
        'l': 0,
        'k': 1,
        'm': 0,
        'f': 0,
        'p': 0,
        's': 0,
        't': 0,
        'w': 0,
        'y': 0,
        'v': 0        
    }
    DEFAULT_GUESS = statistics.median(acid_to_charge.values())
    res = []
    for amino_acid in peptide:
        if amino_acid in acid_to_charge:
            res.append(acid_to_charge[amino_acid])
        else:
            res.append(DEFAULT_GUESS)
    return sum(res)

def get_pks(peptide):
    acid_to_pks = {
        'a': [9.87,2.35],
        'r': [9.09,2.18],
        'n': [8.8,2.02],
        'd': [9.6,1.88],
        'c': [10.78,1.71],
        'e': [9.67,2.19],
        'q': [9.13,2.17],
        'g': [9.6,2.34],
        'h': [8.97,1.78],
        'i': [9.76,2.32],
        'l': [9.6,2.36],
        'k': [10.28,8.9],
        'm': [9.21,2.28],
        'f': [9.24,2.58],
        'p': [10.6,1.99],
        's': [9.15,2.21],
        't': [9.12,2.15],
        'w': [9.39,2.38],
        'y': [9.11,2.2],
        'v': [9.72,2.29]        
    }
    DEFAULT_GUESS = statistics.median(sum(v) for v in acid_to_pks.values())
    res = []
    for amino_acid in peptide:
        if amino_acid in acid_to_pks:
            res.append(sum(acid_to_pks[amino_acid]))
        else:
            res.append(DEFAULT_GUESS)
    return int(sum(res))

def get_solubility(peptide):
    acid_to_solubility = {
        'a': 15.8,
        'r': 71.8,
        'n': 2.4,
        'd': 0.42,
        'c': 100,
        'e': 0.72,
        'q': 2.6,
        'g': 22.5,
        'h': 4.19,
        'i': 3.36,
        'l': 2.37,
        'k': 100,
        'm': 5.14,
        'f': 2.7,
        'p': 1.54,
        's': 36.2,
        't': 100,
        'w': 1.06,
        'y': 0.038,
        'v': 5.6        
    }
    DEFAULT_GUESS = statistics.median(acid_to_solubility.values())
    res = []
    for amino_acid in peptide:
        if amino_acid in acid_to_solubility:
            res.append(acid_to_solubility[amino_acid])
        else:
            res.append(DEFAULT_GUESS)
    return int(sum(res))

# Add in here to add a local feature
def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng, k, gapfactor, log=False):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue

    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    #! edited so that cannot predict excess, taking into account k-1 window on each side
    if len(masked_lms) + 1 > num_to_predict: #2*k - 1 + 2*gapfactor > num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      masked_token = None
      original_token = tokens[index]

      hydrophobicity = get_hydrophobicity(original_token) if FLAGS.do_hydro else 0
      charge = get_charge(original_token) if FLAGS.do_charge else 0
      pks = get_pks(original_token) if FLAGS.do_pkss else 0
      solubility = get_solubility(original_token) if FLAGS.do_solubility else 0

      if rng.random() < 0.8:         # 80% of the time, replace with [MASK]
        masked_token = "[MASK]"
      else:
        if rng.random() < 0.5:           # 10% of the time, keep original
          masked_token = tokens[index]
        else: # 10% of the time, replace with random word
          #! TODO: in the future, maybe do something more intelligent than applying
          # the same tandom k-mer as a substitute to all the tokens within the window
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]



      # Masks the selected token and the k-1 neighbour tokens on each side, so that 
      # peptide overlap doesn't trivialize the mask prediction task
      high_index = min(len(cand_indexes) - 1, index + k + gapfactor - 1)
      low_index = max(0, index - k - gapfactor + 1)

      for i in range(low_index, high_index + 1):
        covered_indexes.add(i)
        output_tokens[i] = masked_token
        masked_lms.append(MaskedLmInstance(index=i, label=tokens[i], hydrophobicity=hydrophobicity, charge=charge, pks=pks, solubility=solubility))
   
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  hydrophobicities = []
  charges = []
  pks = []
  solubilities = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)
    hydrophobicities.append(p.hydrophobicity)
    charges.append(p.charge)
    pks.append(p.pks)
    solubilities.append(p.solubility)
  
  return (output_tokens, masked_lm_positions, masked_lm_labels, hydrophobicities, charges, pks, solubilities)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  lost = 0
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      return lost

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1
    lost += 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  tf.compat.v1.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.compat.v1.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng, FLAGS.k, FLAGS.gapfactor)

  output_files = FLAGS.output_file.split(",")
  tf.compat.v1.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.compat.v1.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.compat.v1.app.run()
