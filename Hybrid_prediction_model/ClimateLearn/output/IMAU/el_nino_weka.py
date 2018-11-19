
__author__ = 'ruggero'

import os
import subprocess
import shlex
import numpy as np
#import el_nino_manip as manip
#import ffx
import matplotlib.pyplot as plt
#import el_nino_io as io


def J48(train_set, test_set , CV_folds = 10, C = 0.25, M = 5 ,out_file = "results" , model_file = "tmp.model", model_dir = '', print_feat = 0):
    """
    Method for J48 decision trees algorithm. Training on training set and validating on test set
    :param train_set: path + name of the training set arff file
    :param test_set: path + name of the test set arff file
    :param CV_folds: number of cross validation folds (10 by default is good)
    :param C: Confidence factor for pruning
    :param M: Minimum number of instances per leaf
    :param out_file: path and file with method output printed
    :param model_file: name of the temporary model file
    :param model_dir: name of the directory of the model file
    :param print_feat: which feature to print (0 as default prints the output classified feature)
    :return: returns the parsed  classified result
    """
    command = "java -classpath C:/Users/User/Documents/Thesis/climatelearncheck/ClimateLearn/weka-3-7-12/weka.jar weka.classifiers.trees.J48  -C " + str(C) + " -M " + str(M) + " -x " + str(CV_folds) + " -t " + train_set + ".arff "
    command += " -d " + model_dir + "tmp.model"
    args = shlex.split(command)
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    output = p1.communicate()[0]
    command2 = "java -classpath C:/Users/User/Documents/Thesis/climatelearncheck/ClimateLearn/weka.jar weka.classifiers.trees.J48  -l " + model_dir + "tmp.model -T " + test_set + '.arff -p ' + str(print_feat)
    args2 = shlex.split(command2)
    p2 = subprocess.Popen(args2, stdout=subprocess.PIPE)
    output2 = p2.communicate()[0]
    fil = open(model_dir + out_file + '.arff', "w")
    fil.write(output2)
    fil.close()
    return weka_class_result(model_dir,out_file)

def NN_regression(train_set, test_set , out_file = "results" , model_file = "tmp.model" ,model_dir = '', learn_rate = 0.3 , momentum = 0.2, train_time = 1000 , layers = "a", print_feat = 1):
    """
    Method for Artificial neural network with regression.
    :param train_set: path + name of the training set arff file
    :param test_set:  path + name of the test set arff file
    :param out_file: path and file with method output printed
    :param model_file: name of the temporary model file
    :param model_dir: name of the directory of the model file
    :param learn_rate: learning rate for gradient descend
    :param momentum: momentum of backpropagation
    :param train_time: training time
    :param layers: layer structure of the network, by default "a"
    :param print_feat: which feature to print (0 as default prints the output regression feature)
    :return: returns the parsed  classified result
    """
    command = "java -classpath C:/Users/User/Documents/Thesis/climatelearncheck/ClimateLearn/weka-3-7-12/weka.jar weka.classifiers.functions.MultilayerPerceptron  -L " + str(learn_rate) + " -M " + str(momentum)
    command += " -N " + str(train_time) + " -V 0 -S 0 -E 20 -H " + _layers(layers) + " -t " + train_set + ".arff "
    command += " -d " + model_dir + model_file + ".model"
    args = shlex.split(command)
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    output = p1.communicate()[0]
    command2 = "java -classpath C:/Users/User/Documents/Thesis/climatelearncheck/ClimateLearn/weka-3-7-12/weka.jar weka.classifiers.functions.MultilayerPerceptron  -l " + model_dir + model_file + ".model -T " + test_set + '.arff -p '  + str(print_feat)
    args2 = shlex.split(command2)
    p2 = subprocess.Popen(args2, stdout=subprocess.PIPE)
    output2 = p2.communicate()[0]
    fil = open(model_dir + out_file + '.arff', "w")
    fil.write(output2)
    fil.close()
    return weka_regr_result(model_dir,out_file)

def NN_classification(train_set, test_set , out_file = "results" , model_file = "tmp_model", model_dir = '' , learn_rate = 0.3 , momentum = 0.2, train_time = 500, layers = "a", print_feat = 1):
    """
    Method for Artificial neural network with classification.
    :param train_set: path + name of the training set arff file
    :param test_set:  path + name of the test set arff file
    :param out_file: path and file with method output printed
    :param model_file: name of the temporary model file
    :param model_dir: name of the directory of the model file
    :param learn_rate: learning rate for gradient descend
    :param momentum: momentum of backpropagation
    :param train_time: training time
    :param layers: layer structure of the network, by default "a"
    :param print_feat: which feature to print (0 as default prints the output regression feature)
    :return: returns the parsed  classified result

    """
    command = "java -classpath C:/Users/User/Documents/Thesis/climatelearncheck/ClimateLearn/weka-3-7-12/weka.jar weka.classifiers.functions.MultilayerPerceptron  -L " + str(learn_rate) + " -M " + str(momentum) 
    command += " -N " + str(train_time) + " -V 0 -S 0 -E 20 -H " + layers + " -t " + train_set + ".arff "
    command += " -d " + model_dir + model_file +".model"
    args = shlex.split(command)
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    output = p1.communicate()[0]
    command2 = "java -classpath C:/Users/User/Documents/Thesis/climatelearncheck/ClimateLearn/weka-3-7-12/weka.jar weka.classifiers.functions.MultilayerPerceptron  -l " + model_dir + model_file + ".model -T " + test_set + '.arff -p '  + str(print_feat)
    args2 = shlex.split(command2)
    p2 = subprocess.Popen(args2, stdout=subprocess.PIPE)
    output2 = p2.communicate()[0]
    fil = open(model_dir + out_file + '.arff', "w")
    fil.write(output2)
    fil.close()
    return weka_class_result(model_dir,out_file)

def FFX(dic, p_total = 100, p_train = 70 ,p_test = 30 , pop = []):
    """
    Method for FFX (regression similar to GP). As fitness it uses Normalized root mean squared error (better fitness?)
    FFX installed needed
    :param dic: The dictionary of the data set
    :param p_total: The total amount of data to use for regressions
    :param p_train: percentage of training set
    :param p_test: percentage of test set
    :param pop: features to pop out before applying the method
    :return: ...
    """
    assert p_train + p_test <= 100

    # dividing the domain into train, void and test parts
    length = len(dic[dic.keys()[0]])
    init_train = 0
    fin_train = int(length*float(p_total)/100.0*float(p_train)/100.0)
    init_void = fin_train + 1
    fin_void = int(length*float(p_total)/100.0*float(100.0-p_test)/100.0)
    init_test = fin_void + 1
    fin_test = int(length*float(p_total)/100.0) - 1

    # eliminating some of the features
    new_dic = dic.copy()
    for k in pop:
        new_dic.pop(k, None)

    # Brings event as the last key element (both for regression and classification)
    if manip.is_in_list('ElNino_tau',new_dic.keys()):
        keys = new_dic.keys()
        keys.remove('ElNino_tau')
        keys.append('ElNino_tau')

    dic_train = {}
    dic_test = {}

    for k in new_dic.keys():
        dic_train[k] = np.array([])
        dic_test[k] = np.array([])
    for i in range(init_train,fin_train+1):
        for k in new_dic.keys():
            dic_train[k] = np.append(dic_train[k],new_dic[k][i])
    for i in range(init_test,fin_test+1):
        for k in new_dic.keys():
            dic_test[k] = np.append(dic_test[k],new_dic[k][i])

    keys = sorted(new_dic.keys())
    print keys
    keys.remove('ElNino_tau')
    keys.remove('t0')
    keys.append('ElNino_tau')
    keys.append('t0')


    y_train = dic_train['ElNino_tau']
    x_train = np.zeros(shape=(len(y_train),len(keys)-2))
    for i, t in enumerate(dic_train["t0"]):
            for k, key in enumerate(keys[:-2]):
                x_train[i,k] = dic_train[key][i]


    y_test = dic_test['ElNino_tau']
    x_test = np.zeros(shape=(len(y_test),len(keys)-2))
    for i, t in enumerate(dic_test["t0"]):
            for k, key in enumerate(keys[:-2]):
                x_test[i,k] = dic_test[key][i]


    keys.remove('t0')
    ffx.core.CONSIDER_THRESH = True
    models_ffx = ffx.run(x_train, y_train, x_test, y_test, keys)
    base_fxx = [model.numBases() for model in models_ffx]
    error_fxx = [model.test_nmse for model in models_ffx]
    model = models_ffx[-1]

    new_pred_FFX = np.array([])
    for i in model.simulate(x_test):
        if i >= 0:
            new_pred_FFX = np.append(new_pred_FFX,i)
        else:
            new_pred_FFX = np.append(new_pred_FFX,0.0)

    time = np.array([])
    for i in range(0,len(dic_test['t0'])):
        time = np.append(time,dic_test['t0'][i])

    return time,y_test,new_pred_FFX

def GP(dic, p_total = 100, p_train = 70 ,p_test = 30 , pop = [], working_dir = "",n_gen = 50, n_subpop = 10000):
    """
    Method for genetic programming (working but new features in progress..Java Package required)
    :param dic: The dictionary of the data set
    :param p_total: The total amount of data to use for regressions
    :param p_train: percentage of training set
    :param p_test: percentage of test set
    :param pop: keys to exclude from the method
    :param working_dir: directly wher to write the files used by GP
    :param n_gen: number of generations of GP
    :param n_subpop: number of subpopulation individuals
    :return: returns the regression predicted results
    """
    assert p_train + p_test <= 100

    # dividing the domain into train, void and test parts
    length = len(dic[dic.keys()[0]])
    init_train = 0
    fin_train = int(length*float(p_total)/100.0*float(p_train)/100.0)
    init_void = fin_train + 1
    fin_void = int(length*float(p_total)/100.0*float(100.0-p_test)/100.0)
    init_test = fin_void + 1
    fin_test = int(length*float(p_total)/100.0) - 1

    # eliminating some of the features
    new_dic = dic.copy()
    for k in pop:
        new_dic.pop(k, None)

    # Brings event as the last key element (both for regression and classification)
    if manip.is_in_list('ElNino_tau',new_dic.keys()):
        keys = new_dic.keys()
        keys.remove('ElNino_tau')
        keys.append('ElNino_tau')

    dic_train = {}
    dic_test = {}
    keys.remove('t0')

    for k in keys:
        dic_train[k] = np.array([])
        dic_test[k] = np.array([])
    for i in range(init_train,fin_train+1):
        for k in keys:
            dic_train[k] = np.append(dic_train[k],new_dic[k][i])
    for i in range(init_test,fin_test+1):
        for k in keys:
            dic_test[k] = np.append(dic_test[k],new_dic[k][i])

    # Check how to write better the files
    io.gp_file(dic_train,"train_set",order=keys)
    io.gp_file(dic_test,"test_set",order=keys)


    command = "java -jar /home/ruggero/Desktop/projects/solar/GP_code/Solar/dist/Solar.jar"
    command += " -file /home/ruggero/Desktop/projects/solar/GP_code/Solar/ecj/ec/app/solar/solar_train.params"
    command += " -Xmx500m -Xmx1024m -p eval.problem.training-file=" + working_dir + "train_set.csv "
    command += "-p eval.problem.testing-file=" + working_dir + "test_set.csv"
    args = shlex.split(command)
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    output = p1.communicate()[0]
    #results = GP_result(working_dir + "out.stat")
    #return results
    return None
def LR_regression():
    """
    Method for linear regression...for now not used
    :return:
    """
    return None



def NN_ensemble_2(train_set,test_set,p = 1,layer_1=[2,3],layer_2 = [4,5],number = 4):

    if number > len(layer_1)*len(layer_2):
        number = len(layer_1)*len(layer_2)

    collect = {}
    collect['structure'] = []
    collect['error'] = np.array([])
    collect['dataset'] = []

    result = {}
    res_init = NN_regression(train_set,test_set,print_feat = p,layers=[2,2])
    result['actual'] = np.array(res_init['actual'])
    for i in layer_1:
        for j in layer_2:
            l=[i,j]
            print "running network with layers",l
            #count +=1
            res_part = NN_regression(train_set,test_set,print_feat = p,layers=l)
            error = _NorRMSE(np.array(res_part['actual']),np.array(res_part['predicted']))
            collect['structure'].append(l)
            collect['error']= np.append(collect['error'],error)
            collect['dataset'].append(np.array(res_part['predicted']))


    ind = collect['error'].argsort()


    result['predicted'] = collect['dataset'][ind[0]]
    for i in range(1,len(ind)):
        if i > number:
            break
        result['predicted'] = result['predicted'] + collect['dataset'][ind[i]]


    result['predicted'] = result['predicted']/number
    return result

def NN_ensemble_3(train_set,test_set,p = 1,layer_1=[2,3],layer_2 = [4,5],layer_3 = [4,5],number = 4):

    if number > len(layer_1)*len(layer_2)*len(layer_3):
        number = len(layer_1)*len(layer_2)*len(layer_3)

    collect = {}
    collect['structure'] = []
    collect['error'] = np.array([])
    collect['dataset'] = []

    result = {}
    res_init = NN_regression(train_set,test_set,print_feat = p,layers=[2,2])
    result['actual'] = np.array(res_init['actual'])
    for i in layer_1:
        for j in layer_2:
            for k in layer_3:
                l=[i,j,k]
                print "running network with layers",l
            #count +=1
                res_part = NN_regression(train_set,test_set,print_feat = p,layers=l)
                error = _NorRMSE(np.array(res_part['actual']),np.array(res_part['predicted']))
                collect['structure'].append(l)
                collect['error']= np.append(collect['error'],error)
                collect['dataset'].append(np.array(res_part['predicted']))


    ind = collect['error'].argsort()


    result['predicted'] = collect['dataset'][ind[0]]
    for i in range(1,len(ind)):
        if i > number:
            break
        result['predicted'] = result['predicted'] + collect['dataset'][ind[i]]

    result['predicted'] = result['predicted']/number
    return result




######################### Output options ######################

def weka_class_result(model_dir , out_file):
    """
    Method for parsing weka classification results
    :param model_dir: directory of the output
    :param out_file: file name of the output
    :return: a dictionary with keys "actual" and "predicted"
    """
    results = {}
    results['actual'] = []
    results['predicted'] = []
    with open(model_dir + out_file + '.arff','r') as fin:
        lines = fin.readlines()
    for i in range(5,len(lines) -1):
        linea = splitting(lines[i], ' ')

        if linea[1][2:] == 'no':
            results['actual'].append(0)
        elif linea[1][2:] == 'yes':
            results['actual'].append(1)

        if linea[2][2:] == 'no':
            results['predicted'].append(0)
        elif linea[2][2:] == 'yes':
            results['predicted'].append(1)
    return results

def weka_regr_result(model_dir,out_file):
    """
    Method for parsing weka regression results
    :param model_dir: directory of the output
    :param out_file: file name of the output
    :return: a dictionary with keys "actual" and "predicted"
    """
    results = {}
    results['actual'] = []
    results['predicted'] = []
    with open(model_dir + out_file + '.arff','r') as fin:
        lines = fin.readlines()
    for i in range(5,len(lines) -1):
        linea = splitting(lines[i], ' ')
        results['actual'].append(float(linea[1]))
        results['predicted'].append(float(linea[2]))
    return results

def GP_result(file, csvfs=','):
    """
    Method to read the output of Gp results.
    :param file: The path and file name of the results
    :param csvfs: the separator of the csv file (use default)
    :return: the parsed results in a dictionary with keys "actual" and "predicted"
    """

    outstat = open(file,'r')
    outdata = outstat.readlines()
    outstat.close()

    while 1:
        test = outdata.pop(0)
        if "Testing on:" in test:
            outdata.pop(0)
            break

    #cut out empty last line
    while 1:
        test = outdata.pop()
        if "Fitness" in test:
            break

    #fitness = test[:-1]
    outdata = [x.split('\n')[0] for x in outdata]
    print outdata
    # parse and convert the strings to float using map()
    # To check this part...the second needs to be the error, cannot be the real and predicted
    results = {}
    results['actual'] = map(float,[x.split(csvfs)[0] for x in outdata])
    results['predicted'] = map(float,[x.split(csvfs)[1] for x in outdata])

    return results


######################  Parsing options ######################

def splitting(stringa,spacing = ' '):
    """
    Splitting method...
    :param stringa: the string to be splitted
    :param spacing:  the spacing of interest
    :return: returns a list of the splitted parts
    """
    new_s = []
    for s in stringa.split(spacing):
        if not(s==''):
            new_s.append(s)
    return new_s


def _layers(s = []):
    """
    Util function for neural networks
    :param s: a list of integer determining the number of units each layer. If empty allows
    for default weka structure
    :return: a string readable by weka
    """
    if s == []:
        return "a"
    else:
        layers = " \""
        for i in range(0,len(s)-1):
            layers += str(s[i]) + ", "
        layers += str(s[-1]) + "\" "
        return layers

def _NorRMSE(actual, prediction):
    """
    Calculates the Root mean squared deviation given a vector of of actual values and predicted values
    :param actual: A numpy array of actual values
    :param prediction: A numpy array of predited values
    :return: It returns the numerical value of the RMSD
    """
    error = 0
    assert len(actual) == len(prediction)
    error = np.sqrt(np.mean((actual - prediction) ** 2)) / (np.max(actual) - np.min(actual))
    return error