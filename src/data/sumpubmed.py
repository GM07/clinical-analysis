import os
import concurrent.futures
        
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class SumPubMedFolder:
    path: str
    abv: str

class SumPubMed:

    ABSTRACT_FOLDER = SumPubMedFolder(path='abstract', abv='abstract')
    SHORTER_ABSTRACT_FOLDER = SumPubMedFolder(path='shorter_abstract', abv='abst')    
    ONE_LINE_ABSTRACT_FOLDER = SumPubMedFolder(path='line_text', abv='text')
    TEXT_FOLDER = SumPubMedFolder(path='text', abv='text')

    def __init__(self, path: str):

        self.path = path
        if self.path[-1] != '/':
            self.path += '/'


        self.load()

    def load(self):

        self.number_of_files = len(os.listdir(self.path + self.SHORTER_ABSTRACT_FOLDER.path))

        abstract_files = self.list_abstracts()
        shorter_abstract_files = self.list_shorter_abstracts()
        line_files = self.list_line()
        text_files = self.list_text()

        # assert len(abstract_files) == len(shorter_abstract_files), \
            # f'The number of files in the shorter abstract folder and the abstract folder are not the same ({len(shorter_abstract_files)} != {len(abstract_files)})'
        assert len(shorter_abstract_files) == len(line_files), \
            f'The number of files in the shorter abstract folder and the one line abstract folder are not the same ({len(shorter_abstract_files)} != {len(line_files)})'
        assert len(shorter_abstract_files) == len(text_files), \
            f'The number of files in the shorter abstract folder and the text folder are not the same ({len(shorter_abstract_files)} != {len(text_files)})'
        
        self.shorter_abstracts = self.get_directory_content(self.SHORTER_ABSTRACT_FOLDER, shorter_abstract_files)
        self.abstracts = self.get_directory_content(self.ABSTRACT_FOLDER, abstract_files)
        self.line = self.get_directory_content(self.ONE_LINE_ABSTRACT_FOLDER, line_files)
        self.texts = self.get_directory_content(self.TEXT_FOLDER, text_files)

        # self.data = pd.DataFrame({
            # 'shorter_abstract': self.shorter_abstracts,
            # 'abstract': self.abstracts,
            # 'line': self.line,
            # 'text': self.texts
        # })

    def get_directory_content(self, directory: SumPubMedFolder, files: list[str], max_files: int = 10000):
        paths = [self.path + directory.path + '/' + file for file in files]
        if max_files is None:
            max_files = len(paths)
        dict = {}
        for file in tqdm(paths[:max_files], total=len(paths[:max_files]), desc=f'Loading {directory.abv} files'):
            dict[file] = self.read_file(file)
        return dict

    def read_file(self, file: str):
        return open(file, 'r', buffering=16384).read()
    
    def list_abstracts(self):
        return self.list_files(self.ABSTRACT_FOLDER)
    
    def list_shorter_abstracts(self):
        return self.list_files(self.SHORTER_ABSTRACT_FOLDER)
    
    def list_line(self):
        return self.list_files(self.ONE_LINE_ABSTRACT_FOLDER)
    
    def list_text(self):
        return self.list_files(self.TEXT_FOLDER)

    def list_files(self, folder: SumPubMedFolder):
        return os.listdir(self.path + folder.path)
    
    def get_number_of_files_in_dir(self, dir_path: str):
        return len(os.listdir(self.path + dir_path))
