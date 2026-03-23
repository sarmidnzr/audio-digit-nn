import librosa
from librosa import feature
import numpy as np
from pathlib import Path
import random

try:
    import cupy as cp
    cp.array([1])
except Exception:
    import numpy as cp



class Neural_Net(): # 5 layer Neural Network  I'm initializing to this 60->128->64->32->10 

    def __init__(self):
        self.layer0_neurons = 7424 #amount of input neurons for each layer
        self.layer1_neurons = 512
        self.layer2_neurons = 256
        self.layer3_neurons = 32
        self.outputs = 10

        self.l0_1_weights = self.initializeWeights(7424, 512)
        self.l1_2_weights = self.initializeWeights(512, 256)
        self.l2_3_weights = self.initializeWeights(256, 32)
        self.l3_4_weights = self.initializeWeights(32, 10)
        self.bias = [ cp.zeros((512, 1), dtype=cp.float32), cp.zeros((256, 1), dtype=cp.float32), cp.zeros((32, 1), dtype=cp.float32), cp.zeros((10, 1), dtype=cp.float32)]


        self.layer0_inp = 0 #inputs during inference for a label
        self.layer1_inp = 0
        self.layer2_inp = 0
        self.layer3_inp = 0
        self.loadWeights()


    def initializeWeights(self,prev_layer_outputs, layer_inputs): # Uses He initialization for neural networks based on layers being  60->128->64->10
        #std0_in = np.sqrt(2/layer0_inputs) #60 inputs
        #std1_in = np.sqrt(2/layer1_inputs) #128 inputs
        #std2_in = np.sqrt(2/layer2_inputs) #64 inputs
        #std3_in = np.sqrt(2/layer3_inputs) #32 inputs
        
        std_in = cp.sqrt(2/prev_layer_outputs)
            
        return cp.random.normal(loc=0.0, scale=std_in, size=(layer_inputs, prev_layer_outputs))
    
    def check_saved_weights(self):
        
        if Path("weights.npz").exists():
            self.loadWeights()
        else:
            print('no saved weights')

    def inference(self, initial_features): #arr should be [weight0_arr, weight1_arr, weight2_arr, weight3_arr]
        
        def softmax(final_layer):

            fin_max = cp.max(final_layer, axis=0, keepdims=True)
            e_powered = cp.exp(final_layer-fin_max)
            e_sum = cp.sum(e_powered,axis=0,keepdims=True)
            
            return e_powered/e_sum
        
       
        self.layer0_inp = initial_features

        l0_1result = cp.matmul(self.l0_1_weights, initial_features) + self.bias[0] #output is 128 dimensional
        l0_1ReLU = cp.maximum(0, l0_1result) #ReLU on layer 0
        
        self.layer1_inp = l0_1ReLU
        l1_2result = cp.matmul(self.l1_2_weights, l0_1ReLU) + self.bias[1] #output is 64 dimensional
        l1_2ReLU = cp.maximum(0, l1_2result) #ReLU on layer 1
        
        self.layer2_inp = l1_2ReLU
        l2_3result = cp.matmul(self.l2_3_weights, l1_2ReLU) + self.bias[2]  #output is 32 dimensional
        l2_3ReLU = cp.maximum(0, l2_3result) #ReLU on layer 2
        
        self.layer3_inp = l2_3ReLU
        l3_4result = cp.matmul(self.l3_4_weights, l2_3ReLU) + self.bias[3] #output is 10 dimensional

        soft = softmax(l3_4result)

        if soft.shape[1] == 1:
            return (soft, self.label_guessed(softmax_in=soft))

        return (soft, 0) #10 dimensional vector 0-9

    def label_guessed(self, softmax_in, val=0):
        
        if val==1:
            return cp.argmax(softmax_in, axis=0).astype(cp.int32).get().tolist()
        
        return int(cp.argmax(softmax_in, axis=0).get()[0])

    
    def backprop(self, fin_output, correct_label_ind, learning_rate):
        
        #output gradient
        batchsize = fin_output[0].shape[1]
        
        #gvec = cp.zeros((self.outputs,1), dtype=cp.float32)
        gstack = cp.zeros_like(fin_output[0])
        gstack[correct_label_ind, cp.arange(batchsize)] = 1

        error_signal_4 = fin_output[0] - gstack # error signal pL/pZ
        grad_weights_4 = cp.matmul(error_signal_4, cp.transpose(self.layer3_inp)) #pdL/pDW
        grad_weights_4/=batchsize
        
        self.bias[3] = self.bias[3] - learning_rate* cp.mean(error_signal_4,axis=1,keepdims=True)

        #prev layer grad
        error_signal_3 = cp.matmul(cp.transpose(self.l3_4_weights),error_signal_4) * Neural_Net.activation_derivative(self.layer3_inp)
        grad_weights_3 = cp.matmul(error_signal_3,cp.transpose(self.layer2_inp))
        grad_weights_3/=batchsize

        self.bias[2] = self.bias[2] - learning_rate* cp.mean(error_signal_3,axis=1,keepdims=True)

        #prev layer grad
        error_signal_2 = cp.matmul(cp.transpose(self.l2_3_weights),error_signal_3) * Neural_Net.activation_derivative(self.layer2_inp)
        grad_weights_2 = cp.matmul(error_signal_2, cp.transpose(self.layer1_inp))
        grad_weights_2/=batchsize

        self.bias[1] = self.bias[1] - learning_rate*cp.mean(error_signal_2,axis=1,keepdims=True)

        #prev layer grad
        error_signal_1 = cp.matmul(cp.transpose(self.l1_2_weights),error_signal_2) * Neural_Net.activation_derivative(self.layer1_inp)
        grad_weights_1 = cp.matmul(error_signal_1, cp.transpose(self.layer0_inp))
        grad_weights_1/=batchsize

        self.bias[0] = self.bias[0] - learning_rate*cp.mean(error_signal_1,axis=1,keepdims=True)
        
        self.l3_4_weights -=  learning_rate*(grad_weights_4+0.0001*self.l3_4_weights)
        self.l2_3_weights -=  learning_rate*(grad_weights_3+0.0001*self.l2_3_weights)
        self.l1_2_weights -=  learning_rate*(grad_weights_2+0.0001*self.l1_2_weights)
        self.l0_1_weights -=  learning_rate*(grad_weights_1+0.0001*self.l0_1_weights)

    def activation_derivative(value):
        
        return (value>0).astype(cp.float32)

    def train(self, batch=100, epoch=300, learning_rate=0.001):
        all_batch_matrices = create_batch_matrices(batch)
        lr = learning_rate
        prev_val = 0
        rep_rate = 0
        correct = 0
        totaltrain =0
        for i in range(epoch):
            random.shuffle(all_batch_matrices)
            print('Training '+ str((i/epoch)*100)+ '% Complete: Epoch '+ str(i))

            for j in range(len(all_batch_matrices)):
                safe_inference = self.inference(initial_features=all_batch_matrices[j][0])
                self.backprop(safe_inference, all_batch_matrices[j][1], lr)

                guessedlabs = cp.argmax(safe_inference[0], axis=0)
                correct += cp.sum(guessedlabs == all_batch_matrices[j][1]).item()
                totaltrain += all_batch_matrices[j][1].size
            
            trainaccuracy = (correct/totaltrain)*100
            valaccuracy = self.runval()*100

            if valaccuracy-0.7 < prev_val < valaccuracy+0.7: # learning decay
                rep_rate+=1
            else: 
                rep_rate = 0
            
            if rep_rate > 10:
                 lr*=0.4
                 rep_rate = 0
            prev_val = valaccuracy
            print('Training Accuracy is: ' + str(trainaccuracy)+ '% Correct')
            print('Validation is '+str(valaccuracy) + '% Correct')
            correct = 0
            totaltrain = 0

        self.saveWeights()

    def extract_labels(self, path=r'/common/home/sn887/audio-digit-nn/wav_audio_files/val', saveto='/common/home/sn887/audio-digit-nn/valdata.npz'): #val =1 train =0
        labs = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}
        files = get_all_paths(path_to_dir=path)
        labels = cp.asarray([labs[f.parent.name] for f in files])
        print("running")
        cp.savez(saveto ,labels=labels)
        print("done")
            

    def runval(self):
        files = get_all_paths(path_to_dir=r'/common/home/sn887/audio-digit-nn/wav_audio_files/val')
        labels = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}

        tlist = [extract_normalized_features(file_path=f) for f in files]
        x = cp.concatenate(tlist,axis=1)
        file = cp.load('valdata.npz')
        reallabels = file["labels"].tolist()
        guessedlabs = cp.argmax(self.inference(initial_features=x)[0],axis=0).get().tolist()

        return sum(z == y for z,y in zip(reallabels,guessedlabs))/len(reallabels)
    
    def saveWeights(self): #saves weights and biases from training
        
        cp.savez('weights.npz', W1=self.l0_1_weights, W2=self.l1_2_weights, W3=self.l2_3_weights, W4=self.l3_4_weights, b1=self.bias[0], b2=self.bias[1], b3=self.bias[2], b4=self.bias[3])

    def loadWeights(self):
        data = cp.load(r'weights.npz')

        self.l0_1_weights = data["W1"]
        self.l1_2_weights = data["W2"]
        self.l2_3_weights = data["W3"]
        self.l3_4_weights = data["W4"]

        self.bias[0] = data["b1"]
        self.bias[1] = data["b2"]
        self.bias[2] = data["b3"]
        self.bias[3] = data["b4"]
    
    #app architecture
    #--------------------------------------------------------------------------------------
    def app_label_guessed(self, softmax_in, val=0):
        
        result = np.argmax(softmax_in, axis=0)
        
        return int(result[0])
    
    def app_loadWeights(self):
        data = np.load(r'weights.npz')

        self.l0_1_weights = data["W1"]
        self.l1_2_weights = data["W2"]
        self.l2_3_weights = data["W3"]
        self.l3_4_weights = data["W4"]

        self.bias[0] = data["b1"]
        self.bias[1] = data["b2"]
        self.bias[2] = data["b3"]
        self.bias[3] = data["b4"]

    def app_inference(self, initial_features): #arr should be [weight0_arr, weight1_arr, weight2_arr, weight3_arr]
            
            def softmax(final_layer):

                fin_max = np.max(final_layer, axis=0, keepdims=True)
                e_powered = np.exp(final_layer-fin_max)
                e_sum = np.sum(e_powered,axis=0,keepdims=True)
                
                return e_powered/e_sum
            
        
            self.layer0_inp = initial_features

            l0_1result = np.matmul(self.l0_1_weights, initial_features) + self.bias[0] #output is 128 dimensional
            l0_1ReLU = np.maximum(0, l0_1result) #ReLU on layer 0
            
            self.layer1_inp = l0_1ReLU
            l1_2result = np.matmul(self.l1_2_weights, l0_1ReLU) + self.bias[1] #output is 64 dimensional
            l1_2ReLU = np.maximum(0, l1_2result) #ReLU on layer 1
            
            self.layer2_inp = l1_2ReLU
            l2_3result = np.matmul(self.l2_3_weights, l1_2ReLU) + self.bias[2]  #output is 32 dimensional
            l2_3ReLU = np.maximum(0, l2_3result) #ReLU on layer 2
            
            self.layer3_inp = l2_3ReLU
            l3_4result = np.matmul(self.l3_4_weights, l2_3ReLU) + self.bias[3] #output is 10 dimensional

            soft = softmax(l3_4result)

            if soft.shape[1] == 1:
                return (soft, self.app_label_guessed(softmax_in=soft))

            return (soft, 0) #10 dimensional vector 0-9

def app_extract_normalized_features(file_path=None, live_audio=None):

        stats = np.load(r"stats.npz")
        mean, stdev = stats["mean"], stats["stdev"]
        if file_path is not None: #for app
            x = extract_features(file_path).astype(np.float32)     
            x = (x - mean) / stdev             
            
            return x
        else: #for live audio
            x = extract_features(raw=live_audio).astype(np.float32)     
            x = (x - mean) / stdev             
            
            return x
        
 #-------------------------------------------------------------------------------------------------
        
def collect_and_save_features(filepath=r'/common/home/sn887/audio-digit-nn/wav_audio_files/train'): #speeds up time of training by just storing the features instead of computing them every single epoch
    
    labels = {"zero":'savedfeatures/zero/',"one":'savedfeatures/one/',"two":'savedfeatures/two/',"three":'savedfeatures/three/',"four":'savedfeatures/four/',"five":'savedfeatures/five/',"six":'savedfeatures/six/',"seven":'savedfeatures/seven/',"eight":'savedfeatures/eight/',"nine":'savedfeatures/nine/'}

    stuff = get_all_paths()
    for i in range(len(stuff)):
        cp.savez(labels[stuff[i].parent.name]+stuff[i].stem+'.npz', feature=extract_normalized_features(stuff[i]))

def extract_features(file=None, raw=None):

            try:
                if file is not None:
                    y,sr = librosa.load(file, sr=16000) # sampling rate default to 16000 Hz
                    y_fixed = librosa.util.fix_length(y,size=sr)

                    feats = feature.melspectrogram(y=y_fixed, sr=sr, hop_length = 160, win_length = 400, n_fft=400, n_mels=64, fmin=20,fmax=8000)
                    logdb = librosa.power_to_db(feats, ref=np.max)
                    return split_feats_into_segments(cp.asarray(logdb))

                else:
                    sr = 16000
                    norm_raw = raw/max(0.000001, float(np.max(np.abs(raw))))
                    ytrim, irsr = librosa.effects.trim(norm_raw,top_db=25)
                    y_fixed = librosa.util.fix_length(ytrim,size=sr)

                    feats = feature.melspectrogram(y=y_fixed, sr=sr, hop_length = 160, win_length = 400, n_fft=400, n_mels=64, fmin=20,fmax=8000)
                    logdb = librosa.power_to_db(feats, ref=np.max)
                    return split_feats_into_segments(cp.asarray(logdb))
                    


            except Exception as e:
                print(e)
    
def extract_normalized_features(file_path=None, live_audio=None):

        stats = cp.load(r"stats.npz")
        mean, stdev = stats["mean"], stats["stdev"]
        if file_path is not None: #for training
            x = extract_features(file_path).astype(cp.float32)     
            x = (x - mean) / stdev             
            
            return x
        else: #for live audio
            x = extract_features(raw=live_audio).astype(cp.float32)     
            x = (x - mean) / stdev             
            
            return x

def split_feats_into_segments(matrix):
                submatrix1 = matrix[:,0:20]
                submatrix2 = matrix[:,20:40]
                submatrix3 = matrix[:,40:60]
                submatrix4 = matrix[:,60:80]
                submatrix5 = matrix[:,80:101]

                global_mean = cp.mean(matrix, axis=1, keepdims=True)
                global_std = cp.std(matrix, axis=1, keepdims=True)

                sm1_std = cp.std(submatrix1, axis=1, keepdims=True)
                sm2_std = cp.std(submatrix2, axis=1, keepdims=True)
                sm3_std = cp.std(submatrix3, axis=1, keepdims=True)
                sm4_std = cp.std(submatrix4, axis=1, keepdims=True)
                sm5_std = cp.std(submatrix5, axis=1, keepdims=True)

                sm1_mean = cp.mean(submatrix1, axis=1, keepdims=True)
                sm2_mean = cp.mean(submatrix2, axis=1, keepdims=True)
                sm3_mean = cp.mean(submatrix3, axis=1, keepdims=True)
                sm4_mean = cp.mean(submatrix4, axis=1, keepdims=True)
                sm5_mean = cp.mean(submatrix5, axis=1, keepdims=True)

                percentiles = cp.percentile(matrix, (10,50,90), axis=1)
                p_10th = percentiles[0][:, None] 
                p_50th = percentiles[1][:, None] 
                p_90th = percentiles[2][:, None] 


                return (cp.concatenate((global_mean, global_std, sm1_mean, sm1_std, sm2_mean, sm2_std, sm3_mean, sm3_std, sm4_mean, sm4_std, sm5_mean, sm5_std, p_10th, p_50th, p_90th, matrix), axis=1)).reshape(-1,1)



def normalize_features():

        file_names = get_all_paths()
        cp_matrix = cp.concatenate([extract_features(file=x) for x in file_names], axis=1) #(numfiles,60) matrix
        
        mean =  cp.mean(cp_matrix, axis=1, keepdims=True)       
        stdev = cp.std(cp_matrix, axis=1, keepdims=True) 
        
        cp.savez("stats.npz", mean=mean, stdev=stdev)

        return mean, stdev

def get_stats():
    stats = cp.load("stats.npz")
    mean, stdev = stats["mean"], stats["stdev"]
    print('mean')
    print(mean)
    print('standard deviation')
    print(stdev)
    
def get_all_paths(path_to_dir=r'/common/home/sn887/audio-digit-nn/wav_audio_files/train'): #path to all wav audio files in train dir
        
        ap = Path(path_to_dir)
        listof_folders = [Path(x) for x in ap.iterdir()]
        list_of_all_files = [x for f in listof_folders for x in f.iterdir()]

        return list_of_all_files

def create_batch_matrices(batchsize): #create batch matrices and labels
    labels = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}

    stuff = get_all_paths()
    batches = []
    for j in range(0, len(stuff), batchsize):
        files = stuff[j:j+batchsize]
        feats = [cp.load(str(Path('/common/home/sn887/audio-digit-nn/savedfeatures/'+f.parent.name)/Path(f.stem+'.npz')))['feature'] for f in files]   
        labs = [labels[f.parent.name] for f in files]                  

        x = cp.concatenate(feats, axis=1)                                  
        y = cp.asarray(labs, dtype=cp.int32).reshape(-1)                            
        batches.append((x, y))

    return batches
    

if __name__ == "__main__":
    tim_net = Neural_Net()
    tim_net.train(batch=40,epoch=400)

    
    
    
    

    