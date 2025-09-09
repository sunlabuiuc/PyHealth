"""Social determinants of health (SDoH) classification.

"""
__author__ = 'Paul Landes'
from typing import Dict, Any, Set, ClassVar
from dataclasses import dataclass, field
import re
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import Pipeline
from peft import PeftModelForCausalLM
from pyhealth.models.base_model import BaseModel


# the prompt and role used to supervised-fine tune the model
_PROMPT: str = """\
Classify sentences for social determinants of health (SDOH).

Definitions SDOHs are given with labels in back ticks:

* `housing`: The status of a patient’s housing is a critical SDOH, known to affect the outcome of treatment.

* `transportation`: This SDOH pertains to a patient’s inability to get to/from their healthcare visits.

* `relationship`: Whether or not a patient is in a partnered relationship is an abundant SDOH in the clinical notes.

* `parent`: This SDOH should be used for descriptions of a patient being a parent to at least one child who is a minor (under the age of 18 years old).

* `employment`: This SDOH pertains to expressions of a patient’s employment status. A sentence should be annotated as an Employment Status SDOH if it expresses if the patient is employed (a paid job), unemployed, retired, or a current student.

* `support`: This SDOH is a sentence describes a patient that is actively receiving care support, such as emotional, health, financial support.  This support comes from family and friends but not health care professionals.

* `-`: If no SDOH is found.

Classify sentences for social determinants of health (SDOH) as a list labels in three back ticks. The sentence can be a member of multiple classes so output the labels that are mostly likely to be present.

### Sentence: {sent}
### SDOH labels:"""


@dataclass
class SdohClassifier(BaseModel):
    """This predicts sentence level social determinants of health (SDoH) as a
    multi-label classification from clinical text.  The model was trained from
    the MIMIC-III derived dataset from `Guevara et al. (2024)`_.

    **Important**: The :obj:`api_key` needs to be populated if the ``Llama 3.1
    8B Instruct`` (or the setting of :obj:`base_model_id`) has not yet been
    downloaded.


    Example::
        >>> from pyhealth.models import SdohClassifier
        >>> sdoh = SdohClassifier()
        >>> sent = 'Pt is homeless and has no car and has no parents or support'
        >>> print(sdoh.predict(sent))
        >>> {'housing', 'transportation'}


    Citation:

      `Guevara et al. (2024)`_ Large language models to identify social determinants of
    health in electronic health records

    .. _Guevara et al. (2024): https://www.nature.com/articles/s41746-023-00970-0

    """
    _ROLE: ClassVar[str] = 'You are a social determinants of health (SDOH) classifier.'
    _LABELS: ClassVar[str] = 'transportation housing relationship employment support parent'.split()

    api_key: str = field(default=None)
    """The API token that starts with ``tf_`` needed to download the Llama
    model.

    """
    base_model_id: str = field(default='meta-llama/Llama-3.1-8B-Instruct')
    """The base model ID, which probably should not be modified."""

    adapter_model_id: str = field(default='plandes/sdoh-llama-3-1-8b')
    """The LoRA adapter model ID, which probably should not be modified."""

    def _parse_response(self, text: str) -> Set[str]:
        """Parse the LLM response (also used in the unit test case).."""
        res_regs = (re.compile(r'(?:.*?`([a-z,` ]{3,}`))', re.DOTALL),
                    re.compile(r'.*?[`#-]([a-z, \t\n\r]{3,}?)[`-].*', re.DOTALL))
        matched: str = ''
        for pat in res_regs:
            m: re.Match = pat.match(text)
            if m is not None:
                matched = m.group(1)
                break
        return set(filter(lambda s: matched.find(s) > -1, self._LABELS))

    def _mod_ignore_check_type(self):
        from transformers.pipelines.text_generation import TextGenerationPipeline

        def noop(*args, **kwargs):
            pass

        TextGenerationPipeline.check_model_type = noop

    def _get_pipeline(self) -> Pipeline:
        """Create the text generation pipeline.  The output is parsed by
        :meth:`_parse_response`."""
        if not hasattr(self, '_pipeline'):
            params: Dict[str, Any] = {}
            if self.api_key is not None:
                params['token'] = self.api_key
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id, **params)
            model = PeftModelForCausalLM.from_pretrained(
                base_model, self.adapter_model_id, **params)
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id, **params)
            # suppress bogus error logging message under transformers 4.53
            # https://github.com/huggingface/transformers/issues/29395
            self._mod_ignore_check_type()
            # create a pipeline for inferencing
            self._pipeline = transformers.pipeline(
                'text-generation',
                framework='pt',
                model=model,
                tokenizer=tokenizer,
                model_kwargs={'torch_dtype': torch.bfloat16},
                device_map='auto')
        return self._pipeline

    def predict(self, sent: str) -> Set[str]:
        """Predict the SDoH labels of ``sent`` (see class docs).

        :param sent: the sentence text used for prediction

        :return: the SDoH labels predicted by the model

        """
        # prompt used by the chat template
        messages = [
            {'role': 'system', 'content': self._ROLE},
            {'role': 'user', 'content': _PROMPT.format(sent=sent)}]

        pipeline: Pipeline = self._get_pipeline()
        # inference the LLM
        outputs = pipeline(
            messages,
            max_new_tokens=512,
            eos_token_id=[
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
            ],
            pad_token_id=pipeline.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.01)

        # the textual LLM output
        output = outputs[0]['generated_text'][-1]['content']
        return self._parse_response(output)
