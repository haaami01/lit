# Copyright 2020 Google LLC
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
# ==============================================================================
# Lint as: python3
"""Tests for lit_nlp.components.gradient_maps."""

import random
from typing import Dict, Iterable, List, Optional

from absl import logging
from absl.testing import absltest
import attr
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.components import tcav
from lit_nlp.lib import utils
import numpy as np
import tensorflow as tf
import transformers


JsonDict = lit_types.JsonDict
Spec = lit_types.Spec

BERT_TINY_PATH = '/cns/od-d/home/iftenney/academic/rs=6.3/lit/models/hf/finetuned/bert-tiny/sst2'


def _from_pretrained(cls, *args, **kw):
  """Load a transformers model in TF2, with fallback to PyTorch weights."""
  try:
    return cls.from_pretrained(*args, **kw)
  except OSError as e:
    logging.warning('Caught OSError loading model: %s', e)
    logging.warning(
        'Re-trying to convert from PyTorch checkpoint (from_pt=True)')
    return cls.from_pretrained(*args, from_pt=True, **kw)


@attr.s(auto_attribs=True, kw_only=True)
class TestGlueModelConfig(object):
  """Config options for a GlueModel."""
  # Preprocessing options
  max_seq_length: int = 128
  inference_batch_size: int = 32
  # Input options
  text_a_name: str = 'sentence1'
  text_b_name: Optional[str] = 'sentence2'  # set to None for single-segment
  label_name: str = 'label'
  # Output options
  labels: Optional[List[str]] = None  # set to None for regression
  null_label_idx: Optional[int] = None
  compute_grads: bool = True  # if True, compute and return gradients.


class TestGlueModel(lit_model.Model):
  """GLUE model, using Keras/TF2 and Huggingface Transformers.
  """

  def __init__(self,
               model_name_or_path=BERT_TINY_PATH,
               **config_kw):
    self.config = TestGlueModelConfig(**config_kw)
    self._load_model(model_name_or_path)

  def _load_model(self, model_name_or_path):
    """Load model. Can be overridden for testing."""
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path)
    model_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(self.config.labels),
        return_dict=False,  # default for training; overridden for predict
    )
    self.model = _from_pretrained(
        transformers.TFAutoModelForSequenceClassification,
        model_name_or_path,
        config=model_config)

  def _preprocess(self, inputs: Iterable[JsonDict]) -> Dict[str, tf.Tensor]:
    if self.config.text_b_name:
      segments = [(ex[self.config.text_a_name], ex[self.config.text_b_name])
                  for ex in inputs]
    else:
      segments = [ex[self.config.text_a_name] for ex in inputs]
    encoded_input = self.tokenizer.batch_encode_plus(
        segments,
        return_tensors='tf',
        add_special_tokens=True,
        max_length=self.config.max_seq_length,
        padding='longest',
        truncation='longest_first')
    return encoded_input

  def _segment_slicers(self, tokens: List[str]):
    """Slicers along the tokens dimension for each segment.

    For tokens ['[CLS]', a0, a1, ..., '[SEP]', b0, b1, ..., '[SEP]'],
    we want to get the slices [a0, a1, ...] and [b0, b1, ...]

    Args:
      tokens: <string>[num_tokens], including special tokens

    Returns:
      (slicer_a, slicer_b), slice objects
    """
    try:
      split_point = tokens.index(self.tokenizer.sep_token)
    except ValueError:
      split_point = len(tokens) - 1
    slicer_a = slice(1, split_point)  # start after [CLS]
    slicer_b = slice(split_point + 1, len(tokens) - 1)  # end before last [SEP]
    return slicer_a, slicer_b

  def _postprocess(self, output: Dict[str, np.ndarray]):
    """Per-example postprocessing, on NumPy output."""
    ntok = output.pop('ntok')
    output['tokens'] = self.tokenizer.convert_ids_to_tokens(
        output.pop('input_ids')[:ntok])

    # Tokens for each segment, individually.
    slicer_a, slicer_b = self._segment_slicers(output['tokens'])
    output['tokens_' + self.config.text_a_name] = output['tokens'][slicer_a]
    if self.config.text_b_name:
      output['tokens_' + self.config.text_b_name] = output['tokens'][slicer_b]

    # Embeddings for each segment, individually.
    output['input_embs_' + self.config.text_a_name] = (
        output['input_embs'][slicer_a])
    if self.config.text_b_name:
      output['input_embs_' + self.config.text_b_name] = (
          output['input_embs'][slicer_b])

    # Gradients for each segment, individually.
    if self.config.compute_grads:
      output['token_grad_' +
             self.config.text_a_name] = output['input_emb_grad'][slicer_a]
      if self.config.text_b_name:
        output['token_grad_' +
               self.config.text_b_name] = output['input_emb_grad'][slicer_b]

      # Return the label corresponding to the class index used for gradients.
      output['grad_class'] = self.config.labels[output['grad_class']]

    # Gradients for the CLS token.
    output['cls_grad'] = output['input_emb_grad'][0]

    # Remove 'input_emb_grad' since it's not in the output spec.
    del output['input_emb_grad']

    return output

  ##
  # LIT API implementation
  def max_minibatch_size(self):
    return self.config.inference_batch_size

  def predict_minibatch(self, inputs: Iterable[JsonDict]):
    # Use watch_accessed_variables to save memory by having the tape do nothing
    # if we don't need gradients.
    with tf.GradientTape(
        watch_accessed_variables=self.config.compute_grads) as tape:
      encoded_input = self._preprocess(inputs)

      # Gathers word embeddings from BERT model embedding layer using input ids
      # of the tokens.
      input_ids = encoded_input['input_ids']
      word_embeddings = self.model.bert.embeddings.word_embeddings
      # <tf.float32>[batch_size, num_tokens, emb_size]
      input_embs = tf.gather(word_embeddings, input_ids)

      tape.watch(input_embs)  # Watch input_embs for gradient calculation.

      model_inputs = encoded_input.copy()
      model_inputs['input_ids'] = None
      out: transformers.modeling_tf_outputs.TFSequenceClassifierOutput = (
          self.model(model_inputs, inputs_embeds=input_embs, training=False,
                     output_hidden_states=True, output_attentions=True,
                     return_dict=True))

      batched_outputs = {
          'input_ids': encoded_input['input_ids'],
          'ntok': tf.reduce_sum(encoded_input['attention_mask'], axis=1),
          'cls_emb': out.hidden_states[-1][:, 0],  # last layer, first token
          'input_embs': input_embs,
      }

      # <tf.float32>[batch_size, num_labels]
      batched_outputs['probas'] = tf.nn.softmax(out.logits, axis=-1)

      # If a class for the gradients has been specified in the input,
      # calculate gradients for that class. Otherwise, calculate gradients for
      # the arg_max class.
      arg_max = tf.math.argmax(batched_outputs['probas'], axis=-1).numpy()
      grad_classes = [ex.get('grad_class', arg_max[i]) for (i, ex) in
                      enumerate(inputs)]
      # Convert the class names to indices if needed.
      grad_classes = [self.config.labels.index(label)
                      if isinstance(label, str) else label
                      for label in grad_classes]

      gather_indices = list(enumerate(grad_classes))
      # <tf.float32>[batch_size]
      scalar_pred_for_gradients = tf.gather_nd(batched_outputs['probas'],
                                               gather_indices)
      if self.config.compute_grads:
        batched_outputs['grad_class'] = tf.convert_to_tensor(grad_classes)

    # Request gradients after the tape is run.
    # Note: embs[0] includes position and segment encodings, as well as subword
    # embeddings.
    if self.config.compute_grads:
      # <tf.float32>[batch_size, num_tokens, emb_dim]
      batched_outputs['input_emb_grad'] = tape.gradient(
          scalar_pred_for_gradients, input_embs)

    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Sequence of dicts, one per example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  def input_spec(self) -> Spec:
    ret = {}
    ret[self.config.text_a_name] = lit_types.TextSegment()
    if self.config.text_b_name:
      ret[self.config.text_b_name] = lit_types.TextSegment()

    ret[self.config.label_name] = lit_types.CategoryLabel(
        required=False, vocab=self.config.labels)
    # The input_embs_ and grad_class fields are used for Integrated Gradients.
    ret['input_embs_' + self.config.text_a_name] = lit_types.TokenEmbeddings(
        align='tokens', required=False)
    if self.config.text_b_name:
      ret['input_embs_' + self.config.text_b_name] = lit_types.TokenEmbeddings(
          align='tokens', required=False)
    ret['grad_class'] = lit_types.CategoryLabel(required=False,
                                                vocab=self.config.labels)
    return ret

  def output_spec(self) -> Spec:
    ret = {'tokens': lit_types.Tokens()}
    ret['tokens_' + self.config.text_a_name] = lit_types.Tokens()
    if self.config.text_b_name:
      ret['tokens_' + self.config.text_b_name] = lit_types.Tokens()
    ret['probas'] = lit_types.MulticlassPreds(
        parent=self.config.label_name,
        vocab=self.config.labels,
        null_idx=self.config.null_label_idx)
    ret['cls_emb'] = lit_types.Embeddings()
    ret['cls_grad'] = lit_types.Gradients(grad_for='cls_emb',
                                          grad_target='grad_class')

    # The input_embs_ and grad_class fields are used for Integrated Gradients.
    ret['input_embs_' + self.config.text_a_name] = lit_types.TokenEmbeddings(
        align='tokens_' + self.config.text_a_name)
    if self.config.text_b_name:
      ret['input_embs_' + self.config.text_b_name] = lit_types.TokenEmbeddings(
          align='tokens_' + self.config.text_b_name)

    # Gradients, if requested.
    if self.config.compute_grads:
      ret['grad_class'] = lit_types.CategoryLabel(required=False,
                                                  vocab=self.config.labels)
      ret['token_grad_' + self.config.text_a_name] = lit_types.TokenGradients(
          align='tokens_' + self.config.text_a_name,
          grad_for='input_embs_' + self.config.text_a_name,
          grad_target='grad_class')
      if self.config.text_b_name:
        ret['token_grad_' + self.config.text_b_name] = lit_types.TokenGradients(
            align='tokens_' + self.config.text_b_name,
            grad_for='input_embs_' + self.config.text_b_name,
            grad_target='grad_class')

    return ret


class TestSST2Model(TestGlueModel):
  """Classification model on SST-2."""

  def __init__(self, *args, **kw):
    super().__init__(
        *args,
        text_a_name='segment',
        text_b_name=None,
        labels=['0', '1'],
        null_label_idx=0,
        **kw)


class TCAVTest(absltest.TestCase):

  def setUp(self):
    super(TCAVTest, self).setUp()
    self.tcav = tcav.TCAV()

  def test_hyp_test(self):
    # t-test where p-value != 1.
    scores = [0, 0, 0.5, 0.5, 1, 1]
    random_scores = [3, 5, -8, -100, 0, -90]
    result = self.tcav.hyp_test(scores, random_scores)
    self.assertAlmostEqual(0.1415165926492605, result)

    # t-test where p-value = 1.
    scores = [0.1, 0.13, 0.19, 0.09, 0.12, 0.1]
    random_scores = [0.1, 0.13, 0.19, 0.09, 0.12, 0.1]
    result = self.tcav.hyp_test(scores, random_scores)
    self.assertEqual(1.0, result)

  def test_compute_tcav_score(self):
    dir_deriv_positive_class = [1]
    result = self.tcav.compute_tcav_score(dir_deriv_positive_class)
    self.assertAlmostEqual(1, result)

    dir_deriv_positive_class = [0]
    result = self.tcav.compute_tcav_score(dir_deriv_positive_class)
    self.assertAlmostEqual(0, result)

    dir_deriv_positive_class = [1, -5, 4, 6.5, -3, -2.5, 0, 2]
    result = self.tcav.compute_tcav_score(dir_deriv_positive_class)
    self.assertAlmostEqual(0.5, result)

  def test_tcav(self):
    random.seed(0)  # Sets seed since create_comparison_splits() uses random.

    # Basic test with dummy outputs from the model.
    examples = [
        {'segment': 'a'},
        {'segment': 'b'},
        {'segment': 'c'},
        {'segment': 'd'},
        {'segment': 'e'},
        {'segment': 'f'},
        {'segment': 'g'},
        {'segment': 'h'}]
    indexed_inputs = [
        {
            'id': '1',
            'data': {
                'segment': 'a'
            }
        },
        {
            'id': '2',
            'data': {
                'segment': 'b'
            }
        },
        {
            'id': '3',
            'data': {
                'segment': 'c'
            }
        },
        {
            'id': '4',
            'data': {
                'segment': 'd'
            }
        },
        {
            'id': '5',
            'data': {
                'segment': 'e'
            }
        },
        {
            'id': '6',
            'data': {
                'segment': 'f'
            }
        },
        {
            'id': '7',
            'data': {
                'segment': 'g'
            }
        },
        {
            'id': '8',
            'data': {
                'segment': 'h'
            }
        },
        {
            'id': '9',
            'data': {
                'segment': 'i'
            }
        },
    ]
    model = TestSST2Model()
    dataset_spec = {'segment': lit_types.TextSegment()}
    dataset = lit_dataset.Dataset(dataset_spec, examples)
    config = {
        'concept_set_ids': ['1', '3', '4', '8'],
        'class_to_explain': '1',
        'grad_layer': 'cls_grad',
        'random_state': 0
    }
    result = self.tcav.run_with_metadata(indexed_inputs, model, dataset,
                                         config=config)

    self.assertLen(result, 1)
    expected = {
        'p_val': 0.011726777437423225,
        'random_mean': 0.38888888888888895,
        'result': {
            'score': 0.3333333333333333,
            'cos_sim': [
                0.08869055008913204, -0.12179254220029953, 0.16013095992702112,
                0.24840410149855854, -0.09793296778406307, 0.05165747295795888,
                -0.21578309676266658, -0.06560493260690689, -0.14758925065147974
            ],
            'dot_prods': [
                189.08509630608307, -266.3631716594065, 344.35049825444406,
                547.1449490976728, -211.6639646373804, 112.50243919791063,
                -472.72066220390843, -144.52959856215776, -323.31888146091023
            ],
            'accuracy': 0.6666666666666666
        }
    }

    self.assertDictEqual(expected, result[0])

  def test_tcav_sample_from_positive(self):
    # Tests the case where more concept examples are passed than non-concept
    # examples, so the concept set is sampled from the concept examples.

    random.seed(0)  # Sets seed since create_comparison_splits() uses random.

    # Basic test with dummy outputs from the model.
    examples = [
        {'segment': 'a'},
        {'segment': 'b'},
        {'segment': 'c'},
        {'segment': 'd'},
        {'segment': 'e'},
        {'segment': 'f'},
        {'segment': 'g'},
        {'segment': 'h'}]
    indexed_inputs = [
        {
            'id': '1',
            'data': {
                'segment': 'a'
            }
        },
        {
            'id': '2',
            'data': {
                'segment': 'b'
            }
        },
        {
            'id': '3',
            'data': {
                'segment': 'c'
            }
        },
        {
            'id': '4',
            'data': {
                'segment': 'd'
            }
        },
        {
            'id': '5',
            'data': {
                'segment': 'e'
            }
        },
        {
            'id': '6',
            'data': {
                'segment': 'f'
            }
        },
        {
            'id': '7',
            'data': {
                'segment': 'g'
            }
        },
        {
            'id': '8',
            'data': {
                'segment': 'h'
            }
        },
    ]
    model = TestSST2Model()
    dataset_spec = {'segment': lit_types.TextSegment()}
    dataset = lit_dataset.Dataset(dataset_spec, examples)
    config = {
        'concept_set_ids': ['1', '3', '4', '5', '8'],
        'class_to_explain': '1',
        'grad_layer': 'cls_grad',
        'random_state': 0
    }
    result = self.tcav.run_with_metadata(indexed_inputs, model, dataset,
                                         config=config)

    self.assertLen(result, 1)
    expected = {
        'p_val': 0.300473213229498,
        'random_mean': 0.4,
        'result': {
            'score': 0.8,
            'cos_sim': [
                0.09526739306513442, -0.20441954462232134, 0.05140677457147603,
                0.1498473232206515, 0.06750430155575667, -0.28244333086404405,
                -0.11022451585498444, -0.14478875184053577
            ],
            'dot_prods': [
                152.48776337523717, -335.64998165047126, 82.9958786466712,
                247.80113149741794, 109.53684348045459, -461.8180489832389,
                -181.29095072639183, -239.4781655521131
            ],
            'accuracy': 1.0
        }
    }
    self.assertDictEqual(expected, result[0])

  def test_get_trained_cav(self):
    # 1D inputs.
    x = [[1], [1], [1], [2], [1], [1], [-1], [-1], [-2], [-1], [-1]]
    y = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    cav, accuracy = self.tcav.get_trained_cav(x, y, 0.33, random_state=0)
    np.testing.assert_almost_equal(np.array([[19.08396947]]), cav)
    self.assertAlmostEqual(1.0, accuracy)

    # 2D inputs.
    x = [[-8, 1], [5, 3], [3, 6], [-2, 5], [-8, 10], [10, -5]]
    y = [1, 0, 0, 1, 1, 0]
    cav, accuracy = self.tcav.get_trained_cav(x, y, 0.33, random_state=0)
    np.testing.assert_almost_equal(np.array([[-77.89678676, 9.73709834]]), cav)
    self.assertAlmostEqual(1.0, accuracy)

  def test_compute_local_scores(self):
    cav = np.array([[0, 1]])
    dataset_outputs = [
        {
            'probas': [0.2, 0.8],
            'cls_emb': [5, 12]
        },
        {
            'probas': [0.6, 0.4],
            'cls_emb': [3, 4]
        }
    ]
    cos_sim, dot_prods = self.tcav.compute_local_scores(
        cav, dataset_outputs, 'cls_emb')
    self.assertListEqual([12, 4], dot_prods)
    # Magnitude of cav is 1, magnitude of cls_embs are [13, 5].
    # Cosine similarity is dot / (cav_mag * cls_embs_mag),
    # which is [12/13, 4/5].
    self.assertListEqual([0.9230769230769231, 0.8], cos_sim)

    cav = np.array([[1, 2, 3]])
    dataset_outputs = [
        {
            'probas': [0.2, 0.8],
            'cls_emb': [3, 2, 1]
        },
        {
            'probas': [0.6, 0.4],
            'cls_emb': [1, 2, 0]
        }
    ]
    cos_sim, dot_prods = self.tcav.compute_local_scores(
        cav, dataset_outputs, 'cls_emb')
    self.assertListEqual([10, 5], dot_prods)
    self.assertListEqual([0.7142857142857143, 0.5976143046671968],
                         cos_sim)

  def test_get_dir_derivs(self):
    cav = np.array([[1, 2, 3]])
    dataset_outputs = [
        {
            'probas': [0.2, 0.8],
            'cls_grad': [3, 2, 1],
            'grad_class': '1'
        },
        {
            'probas': [0.6, 0.4],
            'cls_grad': [1, 2, 0],
            'grad_class': '0'
        }
    ]
    # Example where only the first output is in class_to_explain 1.
    dir_derivs = self.tcav.get_dir_derivs(
        cav, dataset_outputs, 'cls_grad', 'grad_class',
        class_to_explain='1')
    self.assertListEqual([10], dir_derivs)

if __name__ == '__main__':
  absltest.main()
