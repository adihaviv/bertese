from difflib import SequenceMatcher
from json import *
from json.decoder import  WHITESPACE

class LamaExample:
    def __init__(self, uuid, source, relation, masked_template, snippet, masked_sentence,
                sub_label, label, valid_for_train=True):
        self.uuid = uuid
        self.source = source
        self.relation = relation
        self.masked_template = masked_template
        self.snippet = snippet
        self.masked_sentence = masked_sentence
        self.sub_label = sub_label
        self.label = label
        self.valid_for_train = valid_for_train
        self.unmasked_sentence = masked_sentence.replace("[MASK]", self.label)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [f"uuid: {self.uuid}\n",
             f"source: {self.source}\n",
             f"relation: {self.relation}\n",
             f"masked_template: {self.masked_template}\n",
             f"snippet: {self.snippet}\n",
             f"masked_sentence: {self.masked_sentence}\n",
             f"sub_label: {self.sub_label}\n",
             f"label: {self.label}\n",
             f"unmasked_sentence: {self.unmasked_sentence}\n",
             f"valid_for_train: {self.valid_for_train}\n",
             f"unmasked_sentence: {self.unmasked_sentence}"
             ]
        return " ".join(l)

    def to_dict(self):
        out = {"uuid": self.uuid,
               "source": self.source,
               "relation": self.relation,
               "masked_template": self.masked_template,
               "masked_sentence": self.masked_sentence,
               "snippet": self.snippet,
               "sub_label": self.sub_label,
               "label": self.label,
               "valid_for_train": self.valid_for_train,
               "unmasked_sentence": self.unmasked_sentence,}
        return out

    @classmethod
    def from_dict(cls, e_dict):
        return cls(e_dict["uuid"], e_dict["source"], e_dict["relation"], e_dict["masked_template"],
                           e_dict["snippet"], e_dict["masked_sentence"], e_dict["sub_label"],
                           e_dict["label"], e_dict["valid_for_train"])


# subclass JSONEncoder
class LamaExampleEncoder(JSONEncoder):
    def default(self, o):
        return o.to_dict()


class LamaExampleDecoder(JSONDecoder):
    def decode(self, s, _w=WHITESPACE.match):
        return super.decode(self, s, _w=_w)


def _longest_substring(str1, str2):
    # initialize SequenceMatcher object with
    # input string
    seqMatch = SequenceMatcher(None, str1, str2)

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)
    match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))

    # return longest substring
    if match.size != 0:
        return str1[match.a: match.a + match.size]
    else:
        return None


def get_sub_label(sentence, sub_label):
    sentence = sentence.lower()
    sub_label = sub_label.lower()

    if sub_label in sentence:
        subject = sub_label
    else:
        subject = _longest_substring(sentence, sub_label)
    return subject


def to_examples_list(data):
    all_examples = []
    for relation in data:
        examples, relation_template = data[relation]
        if len(examples) > 0 and not isinstance(examples[0], LamaExample):
            lama_examples = []
            for e in examples:
                lama_examples.append(LamaExample.from_dict(e))
            examples = lama_examples
        all_examples += examples
    return all_examples
