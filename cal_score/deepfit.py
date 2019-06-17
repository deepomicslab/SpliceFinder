# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:44:04 2019

@author: ruohawang2
"""

import deeplift
from deeplift.conversion import kerasapi_conversion as kc
import numpy as np
np.set_printoptions(threshold=np.inf)
from collections import OrderedDict
from keras.models import load_model


encoded_label1 = np.loadtxt('label2_encoded.txt')
encoded_label1 = encoded_label1.reshape(-1,400, 5)
"""
for j in range(1,15):
    print('======================model:CNN_1D_exclude_transcript_%f==================='%j)
    model = load_model('CNN_1D_exclude_transcript_'+str(j)+'.h5')


    predict = model.predict_classes(encoded_label1).astype('int')
    print(predict)
"""

for j in range(8,9):
    print('======================model:CNN_1D_exclude_transcript_%f======================'%j)
    
    deeplift_model =kc.convert_model_from_saved_files('CNN_1D_exclude_transcript_'+str(j)+'.h5',nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
    
    find_scores_layer_idx = 0
    deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=find_scores_layer_idx, target_layer_idx=-2)
    
    background = OrderedDict([('A', 0.3), ('C', 0.2), ('G', 0.2), ('T', 0.3), ('N', 0)])
    scores = np.array(deeplift_contribs_func(task_idx=1,
                                         input_data_list=[encoded_label1],
                                         input_references_list=[
                                          np.array([background['A'],
                                                    background['C'],
                                                    background['G'],
                                                    background['T'],
                                                    background['N']])[None,None,:]],
                                         batch_size=10,
                                         progress_update=1000))
    

    scores = scores[:,:,:4]
    
    final_score = scores[0,190:210]
    for i in range(1,100):
        final_score = final_score + scores[i,190:210]

    final_score = np.around(final_score, decimals=3)
    print(final_score)
    
"""
    idx = 0
    scores_for_idx = scores[idx]
    original_onehot = onehot_data[idx]
    print(scores_for_idx.shape)
    print(scores_for_idx[:,None].shape)
    scores_for_idx = original_onehot*scores_for_idx[:,None]

    print(scores_for_idx.shape)
   
    from deeplift.visualization import viz_sequence
    viz_sequence.plot_weights(scores[0], subticks_frequency=10, highlight='blue')
    
""" 