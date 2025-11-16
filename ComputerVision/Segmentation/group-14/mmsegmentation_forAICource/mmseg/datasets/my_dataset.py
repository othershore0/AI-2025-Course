from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS
import mmengine.fileio as fileio

@DATASETS.register_module()
class MyDataset(BaseSegDataset):
    METAINFO = dict(   
        classes = ('background', 'crack'), 
        palette = [[0, 0, 0], [255,0,0]])
    def __init__(self,               
        img_suffix='.jpg',              
        seg_map_suffix='.png',              
        reduce_zero_label=False,              
        **kwargs):     
        super(MyDataset, self).__init__(         
                img_suffix=img_suffix,       
                seg_map_suffix=seg_map_suffix,         
                reduce_zero_label=reduce_zero_label,         
                **kwargs)     
        assert fileio.exists(self.data_prefix['img_path'], backend_args=self.backend_args)