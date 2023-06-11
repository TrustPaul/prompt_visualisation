"""DPP Retriever"""

from openicl import DatasetReader
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from openicl.utils.logging import get_logger
from typing import Optional
import tqdm
import numpy as np
import math
from accelerate import Accelerator
from submodlib import FacilityLocationMutualInformationFunction
from submodlib import LogDeterminantMutualInformationFunction
from submodlib import GraphCutMutualInformationFunction
from submodlib import FacilityLocationVariantMutualInformationFunction
from submodlib import ConcaveOverModularFunction
from submodlib_cpp import ConcaveOverModular
logger = get_logger(__name__)


class SubmodRetriever(TopkRetriever):
    """    
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`
        , optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:`model`.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
        seed (:obj:`int`, optional): Seed for the random number generator. (:obj:`random_state` in :obj:`sample_exact_k_dpp` method)
        scale_factor (:obj:`float`, optional): A factor when gets the kernel.
    """
    model = None

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name: Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 candidate_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 seed: Optional[int] = 1,
                 scale_factor: Optional[float] = 0.1,
                 submodular_function:Optional[str] = "facility"
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token,
                         sentence_transformers_model_name, ice_num, index_split, test_split, tokenizer_name, batch_size,
                         accelerator)
        self.candidate_num = candidate_num
        self.seed = seed
        self.scale_factor = scale_factor
        self.submodular_function = submodular_function

    def dpp_search(self):
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")
        rtr_idx_list = [[] for _ in range(len(res_list))]
        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']
            ground_number = 100
            # get TopK results
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = np.array(self.index.search(embed, self.candidate_num)[1][0].tolist())

            # DPP stage
            samples_ids = self.get_kernel(embed, near_ids.tolist(), self.submodular_function)
            rtr_sub_list = [int(near_ids[i]) for i in samples_ids]


            rtr_idx_list[idx] = rtr_sub_list

        return rtr_idx_list

    def retrieve(self):
        return self.dpp_search()

    def get_kernel(self, embed, candidates,sub_function):
        near_reps = np.stack([self.index.index.reconstruct(i) for i in candidates], axis=0)
        # normalize first
        embed = embed 
        near_reps = near_reps 
         # normalize first
        embed = embed / np.linalg.norm(embed)
        near_reps = near_reps / np.linalg.norm(near_reps, keepdims=True, axis=1)
        number_near_reps  = len(near_reps)
        
        if sub_function == 'facility':
            obj = FacilityLocationMutualInformationFunction(n=self.candidate_num, num_queries=1, data=near_reps, 
                                                    queryData=embed, metric="euclidean")
            greedyList = obj.maximize(budget=self.ice_num,optimizer='NaiveGreedy', stopIfZeroGain=False, 
                              stopIfNegativeGain=False, verbose=False)
            greedyXs = [x[0] for x in greedyList]
            
        elif sub_function == 'dpp':
            obj = LogDeterminantMutualInformationFunction(n=self.candidate_num, num_queries=1, data=near_reps, 
                                                    queryData=embed, metric="euclidean",lambdaVal=0.5)
            greedyList = obj.maximize(budget=self.ice_num,optimizer='NaiveGreedy', stopIfZeroGain=False, 
                              stopIfNegativeGain=False, verbose=False)
            greedyXs = [x[0] for x in greedyList]
            
            
        elif sub_function == 'graph_cut':
            obj = GraphCutMutualInformationFunction(n=self.candidate_num, num_queries=1, data=near_reps, 
                                                    queryData=embed, metric="euclidean")
            greedyList = obj.maximize(budget=self.ice_num,optimizer='NaiveGreedy', stopIfZeroGain=False, 
                              stopIfNegativeGain=False, verbose=False)
            greedyXs = [x[0] for x in greedyList]
            
        elif sub_function == 'facility_variant':
            obj = FacilityLocationVariantMutualInformationFunction(n=self.candidate_num, num_queries=1, data=near_reps, 
                                                    queryData=embed, metric="euclidean")
            greedyList = obj.maximize(budget=self.ice_num,optimizer='NaiveGreedy', stopIfZeroGain=False, 
                              stopIfNegativeGain=False, verbose=False)
            greedyXs = [x[0] for x in greedyList]
            
            
        elif sub_function == 'concave':
            obj = ConcaveOverModularFunction(n=self.candidate_num, num_queries=1, data=near_reps, 
                                                    queryData=embed, metric="euclidean")
            greedyList = obj.maximize(budget=self.ice_num,optimizer='NaiveGreedy', stopIfZeroGain=False, 
                              stopIfNegativeGain=False, verbose=False,mode=ConcaveOverModular.logarithmic)
            greedyXs = [x[0] for x in greedyList]
        else:
            print('Not Implemented!')
        


        return greedyXs


