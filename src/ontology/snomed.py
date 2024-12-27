from tqdm import tqdm
import os
import json
from collections import defaultdict, deque
from typing import Dict, List
import logging
import joblib


import owlready2 as owl

logger = logging.getLogger(__name__)

class SClass():
    """
    Class of a snomed concept
    """

    def __init__(self, snomed_class: owl.ThingClass, *args):
        self.owl_class = snomed_class
        self.path = str(snomed_class.iri)
        self.id = self.path[len(Snomed.BASE_PATH):]
        
        # The label attribute includes the parent class
        self.label_parent = self.get_locstr(snomed_class.label.first())
        self.label = self.get_locstr(snomed_class.prefLabel.first())
        if not self.label:
            self.label = self.label_parent
        
        self.all_labels = self.all_labels = snomed_class.label
        self.alt_label = [self.get_locstr(label) for label in snomed_class.altLabel]
        self.definition = [self.get_locstr(defn) for defn in snomed_class.definition]

    def get_locstr(self, locstr, lang="en"):
        if isinstance(locstr, dict):
            return locstr.get(lang, locstr.get('en', ''))
        return locstr

    def __str__(self) -> str:
        out = f'ID : "{self.id}"\n'
        out += f'Label : {self.label}\n'
        # out += f'Alternative label : [{", ".join(self.alt_label) if self.alt_label is not None and len(self.alt_label) > 0 else ""}]\n'
        if len(self.definition) > 0:
            out += f'Definition : [{", ".join(self.definition) if self.definition is not None and len(self.alt_label) > 0 else ""}]\n'
        
        return out

class SRestrictionProperty:
    """
    Class of a snomed restriction property
    """

    def __init__(self, property_type, ids, labels, o_restriction) -> None:
        self.property_type = property_type
        self.ids_to_ids = ids
        self.labels_to_labels = labels
        self.o_restriction = o_restriction

    def get_value(self):
        if self.property_type == 'or':
            return ' or '.join(self.labels_to_labels.values())
        else:
            return ' '.join(self.labels_to_labels.values())
        
    def __str__(self) -> str:
        out = f'Property type : {self.property_type}\n'
        
        for k, v in self.labels_to_labels.items():
            out += f'\t{k} : {v}\n'

        return out

class SClassEncoder(json.JSONEncoder):
        """
        JSON encoder of an `SClass`
        """
        def default(self, o):
            encoded = o.__dict__
            encoded['owl_class'] = ''
            return encoded


class SRestrictionPropertyEncoder(json.JSONEncoder):
        """
        JSON encoder of an `SRestrictionProperty`
        """
        def default(self, o):
            encoded = o.__dict__
            encoded['o_restriction'] = ''
            return encoded

class Snomed:

    BASE_CLASS_ID = '138875005'
    BASE_PATH = 'http://snomed.info/id/'

    ID_TO_CLASS_PATH = 'id_to_classes.snomed'
    PARENTS_OF_PATH = 'parents_of.snomed'
    CHILDREN_OF_PATH = 'children_of.snomed'
    RESTRICTION_PROPERTY_OF_PATH = 'properties_of.snomed'

    DEFAULT_CACHE_NAME = 'cache.snomed'

    PRIMARY_CONCEPTS = ['105590001', '123037004', '123038009', '243796009', '254291000', '260787004', '272379006', '308916002', '362981000', '363787002', '370115009', '373873005', '404684003', '410607006', '419891008', '48176007', '71388002', '78621006', '900000000000441003']

    def __init__(self, path: str, cache_path = './', rebuild=False, nb_classes: int = 366771):
        
        self.cache_path = cache_path
        self.id_to_classes_path = self.cache_path + Snomed.ID_TO_CLASS_PATH
        self.parents_of_path = self.cache_path + Snomed.PARENTS_OF_PATH
        self.children_of_path = self.cache_path + Snomed.CHILDREN_OF_PATH  
        self.restriction_properties_of_path = self.cache_path + Snomed.RESTRICTION_PROPERTY_OF_PATH

        self.id_to_classes: Dict[str, SClass] = {}
        self.parents_of = defaultdict(list)
        self.children_of = defaultdict(list)
        self.restriction_properties_of = defaultdict(list)
        self.not_found = {}

        logger.info(f'Loading ontology from path {path}')
        self.ontology = owl.get_ontology(path).load()
        self.base_class = self.get_class_from_id(Snomed.BASE_CLASS_ID, refetch=True)
        self.nb_classes = nb_classes if nb_classes > -1 else len(list(self.ontology.classes()))

        self.build(rebuild=rebuild)

    def summary(self):
        """
        Summarizes the ontology 
        """
        logger.info(f'Number of classes : {len(self.id_to_classes)}')
        logger.info(f'Number of parents relationships : {len(self.parents_of)}')
        logger.info(f'Number of children relationships : {len(self.children_of)}')
        logger.info(f'Number of properties : {len(self.restriction_properties_of)}')

    def get_subjects(self, concept):
        if type(concept) is owl.class_construct.Restriction:
            return [concept.value]
        elif (type(concept) is owl.class_construct.Or) or (type(concept) is owl.class_construct.And):
            return concept.get_Classes()
        else:
            return [concept]
        
    def build(self, rebuild=False):
        """
        Builds the ontology from the cache or from scratch. By building, we mean create the cache of all classes, all parents
        of classes, all restriction properties of classes and all children of classes.

        Args:
            rebuild: Whether to completely rebuild the cache
        """
        logger.info('Verifying cache...')

        if self.verify_cache() and not rebuild:
            logger.info('Loading cache...')
            self.load()
            return

        logger.info('Could not find cache, rebuilding from scratch...')
        
        for o_class in tqdm(self.ontology.classes(), total=self.nb_classes):
            sclass = SClass(o_class)
            self.id_to_classes[sclass.id] = sclass
        
        for o_class in tqdm(self.ontology.classes(), total=self.nb_classes):
            sclass = SClass(o_class)
            for parent_classes in (self.ontology.get_parents_of(o_class) + o_class.equivalent_to):
                
                for parent_class in self.get_subjects(parent_classes):
                    if isinstance(parent_class, owl.Restriction):
                        parent_value = parent_class.value
                        text_property = ''
                        if isinstance(parent_value, owl.And) or isinstance(parent_value, owl.Or):
                            property_type = 'and'
                            # Remove properties that are not restrictions or object properties
                            classes = list(filter(lambda x: isinstance(x, owl.Restriction)\
                                                and isinstance(x.property, owl.ObjectPropertyClass), \
                                    parent_value.get_Classes()))
                            
                            # Generate labels dictionary 
                            labels = {self.get_class_from_id(x.property._name, refetch=True).label: \
                                        self.get_class_from_id(x.value._name, refetch=True).label for x in classes}
                            
                            # Generate ids dictionary
                            ids = {x.property._name: x.value._name for x in classes}
                        elif isinstance(parent_value, owl.Restriction):
                            property_type = 'res_simple'
                            labels = {self.get_class_from_id(parent_value.property._name, refetch=True).label: \
                                self.get_class_from_id(parent_value.value._name, refetch=True).label}
                            ids = {parent_value.property._name: parent_value.value._name}
                        elif isinstance(parent_value, owl.ThingClass):
                            property_type = 'simple'
                            text_property = self.get_class_from_id(parent_value._name, refetch=True).label
                            ids = {parent_value._name: parent_value._name}
                            labels = {text_property: text_property}
                        else:
                            p = self.get_class_from_id(parent_class.property._name, refetch=True)
                            self.not_found[p.id] = (p, parent_value)
                            continue

                        classes = []
                        classes.extend(list(ids.keys()))
                        classes.extend(list(ids.values()))

                        for c in classes:
                            extracted_sclass = self.get_class_from_id(c, refetch=True)
                            if extracted_sclass is None:
                                logger.warning(f'{c} is none')
                            self.id_to_classes[c] = extracted_sclass
                        
                        property = SRestrictionProperty(property_type=property_type, 
                                                ids=ids, 
                                                labels=labels,
                                                o_restriction=parent_class)
                        self.restriction_properties_of[sclass.id].append(property)
                    else:
                        parent_sclass = SClass(parent_class)
                        self.parents_of[sclass.id].append(parent_sclass.id)
                        self.children_of[parent_sclass.id].append(sclass.id)

    def verify_cache(self):
        return os.path.exists(self.cache_path) and not os.path.isdir(self.cache_path)

    def load(self):
        """
        Loads the cache from the `self.cache_path` attribute.
        """
        cache = joblib.load(self.cache_path)

        id_to_classes_json = json.loads(cache['id_to_classes'])
        for k, v in id_to_classes_json.items():
            if v is None:
                logger.warning('v is none : ', k)
            obj = object.__new__(SClass)
            obj.__dict__ = v
            self.id_to_classes[k] = obj
        
        self.parents_of = json.loads(cache['parents_of'])
        self.children_of = json.loads(cache['children_of'])
        
        restriction_properties_of_json = json.loads(cache['restriction_properties_of'])
        for k, v in restriction_properties_of_json.items():
            for property in v:
                obj = object.__new__(SRestrictionProperty)
                obj.__dict__ = property
                self.restriction_properties_of[k].append(obj)

    def save(self):
        """
        Saves the cache into the path `self.cache_path`
        """
        joblib.dump({
            'id_to_classes': json.dumps(self.id_to_classes, indent=4, cls=SClassEncoder),
            'parents_of': json.dumps(self.parents_of, indent=4, cls=SClassEncoder),
            'children_of': json.dumps(self.children_of, indent=4, cls=SClassEncoder),
            'restriction_properties_of': json.dumps(self.restriction_properties_of, indent=4, cls=SRestrictionPropertyEncoder)
        }, self.cache_path)

    def get_contextual_description_of_id(self, id: str):
        """
        Returns a contextual description of a SNOMED class. This description includes :
        - The name of the class
        - A definition (if present)
        - All possible labels (if present)
        - Ancestors
        - Restriction properties (if present)
        """
        description = ''
        sclass = self.get_class_from_id(id)

        # Medical concept
        description += f'Medical concept: {sclass.label}.\n'

        if len(sclass.definition) > 0:
            description += 'Definitions: ' + '. '.join(sclass.definition) + '.\n'

        if len(sclass.alt_label) > 1:
            # First label is always the name
            labels = sclass.alt_label[1:]
            if not isinstance(labels, list):
                labels = [labels]

            description += 'Synonyms: ' + ', '.join(labels) + '.\n'

        ancestors_ids = self.get_ancestors_of_id(id, return_list=True)
        if len(ancestors_ids) > 1:
            direct_ancestor = self.get_label_from_id(ancestors_ids[0])
            description += f'{sclass.label} is a {direct_ancestor}'
            
            if len(ancestors_ids) > 2:
                description += ', which is a type of '
                description += ', which is a type of '.join(list(map(self.get_label_from_id, ancestors_ids[1:-1])))
            description += '.\n'

        # ancestors = ', '.join(list(map(self.get_label_from_id, ancestors_ids)))

        properties = self.get_restriction_properties_of_id(id)
        if len(properties) > 0:
            description += 'Properties: ' + ', '.join(list(map(lambda x: x.get_value(), properties)))

        return description

    def convert_ids_to_labels(self, ids: List[str], refetch=False):
        return list(map(lambda x: self.get_label_from_id(x, refetch=refetch), ids))

    def convert_ids_to_classes(self, ids: List[str], refetch=False):
        """
        Converts a list of SNOMED ids to SClasses

        Args:
            ids: List of ids to convert to classes

        Returns:
        List of concepts associated to these ids
        """
        classes = []
        for id in ids:
            classes.append(self.get_class_from_id(id, refetch=refetch))
        return classes
    
    def is_id_valid(self, id: str):
        """Returns whether an id is in the ontology or not"""
        return id in self.id_to_classes

    def get_class_from_id(self, id: str, refetch=False) -> SClass:
        """
        Searches the ontology by id (iri). The prefix "http://snomed.info/id/" must be omitted.

        Args:
            id: Id to get the class from
            refetch: Whether to refetch the concept from the ontology or use the cache

        Returns:
        SClass object of the concept linked to the id
        """
        if refetch:
            return SClass(self.__get_class_from_id(id))
        else:
            if id not in self.id_to_classes:
                return SClass(self.__get_class_from_id(id))
            return self.id_to_classes[id]

    def get_label_from_id(self, id: str, refetch=False) -> str:
        """
        Searches the ontology to find the label associated with the id (iri). The prefix "http://snomed.info/id/" must be omitted.

        Args:
            id: Id to get the class from
            refetch: Whether to refetch the concept from the ontology or use the cache

        Returns:
        Label of the concept linked to the id
        """
        return self.get_class_from_id(id, refetch=refetch).label

    def get_restriction_properties_of_id(self, id: str) -> List[SRestrictionProperty]:
        """
        Returns all restriction properties of an id
        """
        if id not in self.restriction_properties_of:
            return []
        else:
            properties = self.restriction_properties_of[id]
            if isinstance(properties, list):
                return properties
            return [properties]

    def get_parents_of_id(self, id: str):
        """
        Returns the direct parents of an id
        """
        if id not in self.parents_of:
            return self.__get_class_from_id(id).parents_of
        parent_ids = self.parents_of[id]
        return self.convert_ids_to_classes(parent_ids)

    def get_ancestors_of_id(self, id: str, return_set=False, return_list=False):
        """
        Returns all ancestors of an id in multiple paths to the root.

        Args:
            id: Id to return the ancestors from
            return_set: Return the ancestors in a set
            return_list: Return the ancestors in a sorted list where the last element is
            the furthest ancestor
        """
        if return_set:
            return set(self.__get_all_ancestors_of_id_as_list(id))
        
        if return_list:
            return self.__get_all_ancestors_of_id_as_list(id)
        
        if id not in self.parents_of:
            return self.__get_class_from_id(id).parents_of

        ancestors = []

        queue = deque([(id, ancestors)])

        while queue:
            current_id, current_list = queue.popleft()
            if current_id in self.parents_of:
                parent_ids = self.parents_of[current_id]
                for parent_id in parent_ids:
                    parent_list = []
                    current_list.append({parent_id: parent_list})
                    queue.append((parent_id, parent_list))

        return ancestors
   
        
    def __get_class_from_id(self, id: str):
        """
        Searches the ontology to get the concept associated with an id
        """
        return self.ontology.search(iri=Snomed.BASE_PATH + id).first()

    def __get_all_ancestors_of_id_as_list(self, id: str):
        """
        Returns all ancestors of an id as a list
        """
        ancestors = list()
        queue = deque([id])

        while queue:
            current_id = queue.popleft()
            if current_id in self.parents_of:
                parent_ids = self.parents_of[current_id]
                for parent_id in parent_ids:
                    if parent_id not in ancestors:
                        ancestors.append(parent_id)
                        queue.append(parent_id)

        return ancestors

    def get_children_of_id(self, id: str, ids_only=False, labels_only=False):
        """
        Returns all direct children of an id

        Args:
            id: Id to return the children of
            ids_only: Whether to return the ids only
            labels_only: Whether to return the labels only
        """
        if id not in self.children_of:
            return []
        child_ids = self.children_of[id]
        if ids_only:
            return child_ids
        if labels_only:
            return list(map(lambda x: x.label, self.convert_ids_to_classes(child_ids)))
        return self.convert_ids_to_classes(child_ids)
