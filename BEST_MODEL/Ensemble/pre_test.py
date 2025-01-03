import pickle
import numpy as np

with open('/home/shixiaohan-toda/Desktop/Challenge/SHI/WavLM_Mamba/Final_result_Mamba_Test.pickle', 'rb') as file:
    Final_result_Mamba = pickle.load(file)
with open('/home/shixiaohan-toda/Desktop/Challenge/SHI/Baseline_Whisper/Final_result_Whisper_Test.pickle', 'rb') as file:
    Final_result_Whisper = pickle.load(file)

print(Final_result_Mamba[0][0])
print(Final_result_Whisper)
for i in range(len(Final_result_Mamba[0])):
    for j in range(len(Final_result_Whisper)):
        if(Final_result_Mamba[0][i]['id'] == Final_result_Whisper[j]['id']):
            Final_result_Mamba[0][i]['Fea_ALL'] = np.hstack((Final_result_Mamba[0][i]['Predict_fea'], Final_result_Whisper[j]['Predict_fea']))
            print(Final_result_Mamba[0][i]['Fea_ALL'].shape)
            print(i)

file = open('/home/shixiaohan-toda/Desktop/Challenge/SHI/Ensemble/Data_test.pickle', 'wb')
pickle.dump(Final_result_Mamba,file)
file.close()


