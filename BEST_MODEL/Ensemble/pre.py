import pickle
import numpy as np

with open('/home/shixiaohan-toda/Desktop/Challenge/SHI/WavLM_Mamba/Final_result_Mamba.pickle', 'rb') as file:
    Final_result_Mamba = pickle.load(file)
with open('/home/shixiaohan-toda/Desktop/Challenge/SHI/Baseline_Whisper/Final_result.pickle', 'rb') as file:
    Final_result_Whisper = pickle.load(file)
with open('/home/shixiaohan-toda/Desktop/Final_result_Correct.pickle', 'rb') as file:
    Final_result_Text = pickle.load(file)

for i in range(len(Final_result_Mamba[0])):
    for j in range(len(Final_result_Whisper[0])):
        if(Final_result_Mamba[0][i]['id'] == Final_result_Whisper[0][j]['id']):
            Final_result_Mamba[0][i]['Fea_ALL'] = np.hstack((Final_result_Mamba[0][i]['Predict_fea'], Final_result_Whisper[0][j]['Predict_fea']))
            #print(Final_result_Mamba[0][i]['Fea_ALL'].shape)
            print(i)

for i in range(len(Final_result_Mamba[0])):
    for j in range(len(Final_result_Text[0])):
        if(Final_result_Mamba[0][i]['id'] == Final_result_Text[0][j]['id']):
            Final_result_Mamba[0][i]['Fea_ALL'] = np.hstack((Final_result_Mamba[0][i]['Fea_ALL'], Final_result_Text[0][j]['Predict_fea']))
            print(Final_result_Mamba[0][i]['Fea_ALL'].shape)
            print(i)

file = open('/home/shixiaohan-toda/Desktop/Challenge/SHI/Ensemble/Data.pickle', 'wb')
pickle.dump(Final_result_Mamba,file)
file.close()


