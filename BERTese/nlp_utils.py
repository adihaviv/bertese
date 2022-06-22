import os
import spacy
import pickle
import random
import re

nlp = spacy.load("en_core_web_sm")


class NEE(object):

    def __init__(self):
        self.entities_dict = {}

    def add_entities(self, snippet):
        doc = nlp(snippet)
        for ent in doc.ents:
            ent_type = ent.label_
            ent_value = clear_text(ent.text)

            if ent_type not in self.entities_dict:
                self.entities_dict[ent_type] = []

            if ent.label_ not in self.entities_dict[ent_type]:
                self.entities_dict[ent_type].append(ent_value)

    def get_random_entity(self, entity_name):
        if entity_name not in self.entities_dict:
            return ""

        return self.entities_dict[entity_name][random.randint(0, len(self.entities_dict[entity_name]))]


def clear_text(s):
    new_text = re.search("[a-zA-Z0-9].*[a-zA-Z0-9]", s)
    if new_text is None:
        return s
    return re.search("[a-zA-Z0-9].*[a-zA-Z0-9]", s).group(0)


def extract_entities(all_data, data_path, nee_file_name, override_nee=False):
    if override_nee or not os.path.exists(os.path.join(data_path, nee_file_name)):
        nee = NEE()
        for relation in all_data:
            print("reading entities for {}".format(relation))
            # TODO: CHECK IF WE SWITCH IT PER RELATION IF IT HELPS
            for e in all_data[relation][0]:
                nee.add_entities(e.snippet)
        print("Finished creating the entities dict")
        pickle.dump(nee, open(os.path.join(data_path, nee_file_name), "wb"))
    else:
        print("Reading nee from pkl file")
        nee = pickle.load(open(os.path.join(data_path, nee_file_name), "rb"))
    return nee
