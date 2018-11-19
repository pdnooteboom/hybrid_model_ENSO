__author__ = 'ruggero'
__author__ = "markus"



import numpy as np
import os
import string
import csv
import scipy.io
import datetime
import time
import el_nino_manip as manip
import scipy.stats as stats

# we need to use version 2.0 of arff (for now the file is in our directory)
import arf


####################### READING ###########################


def read_Net_partial(inputDir,exten = '.dat'):
    """
    Reads files from a directory with a given extension and write them in a dictionary. Used for Armin dataset
    It allows for nan data
    :param inputDir: The input directory
    :param exten: The extension of the file
    :return: returns the dictionary of the data set
    """
    if os.path.isdir(inputDir) == True:
        dic = {}
        dic['date_time'] = np.array([])
        file_num = 0
        for f in os.listdir(inputDir) :			
            extension = os.path.splitext(f)[1]
            inFile = inputDir+f
            if extension == exten:
                file_num += 1
                feat_name = f[0:len(f)-len(extension)]
                dic[feat_name] = np.array([])
                with open(inFile, 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    ele = -1
                    for row in reader:
                        if len(row) != 0:
                            dt =(float)(row[0].split("\t")[0])
                            try:
							    value =(float)(row[0].split("\t")[1])
                            except: # should no happen
                                v = float('nan')
                            if file_num == 1:
                                dic['date_time'] = np.append(dic['date_time'],dt)
                                dic[feat_name] = np.append(dic[feat_name],value)
                            else:
                                if manip.is_in_list(dt,dic['date_time']):
                                    dic[feat_name] = np.append(dic[feat_name],value)
                                else: 
                                    dic['date_time'] = np.append(dic['date_time'],dt)
                                    dic[feat_name] = np.append(dic[feat_name],value)

        return dic
    else:
        print "Wrong input directory provided. Exiting!"
        exit(1)
        return 0

def read_ElNino(file_name,sep = '\t'):
    """
    Method for Elnino3.4
    :param file_name: name of the file with the path
    :return: returns the dictionary of the data
    """
    dic = {}
    dic['date_time'] = np.array([])
    dic['ElNino'] = np.array([])
    data = csv.reader(open(file_name),delimiter=sep, quotechar='|')
    for row in data:
        n = 0
        for j in row:
            if n == 0:
                dic['date_time'] = np.append(dic['date_time'],float(j))
                n = 1
            else:
                dic['ElNino'] = np.append(dic['ElNino'],float(j))
    return dic

def read_Qing_Alexis(file_net,file_clos,file_nino):
    """
    Method to read the network data and elnino data from Utrecht. Bugged (it works specifically for data having
    a common date_time)
    :param file_net: path and namefile for network data
    :param file_nino: path and name file for nino3 data
    :return: returns the dictionary
    """
    data = np.loadtxt(file_net)
    #data_clos = np.loadtxt(file_clos)
    meanDegree = data.mean(1)
    varianceDegree = data.var(1)
    skewDegree = stats.skew(data,1)
    kurtDegree = stats.kurtosis(data,1)

    #meanDegree_c = data_clos.mean(1)
    #varianceDegree_c = data_clos.var(1)
    #skewDegree_c = stats.skew(data_clos,1)
    #kurtDegree_c = stats.kurtosis(data_clos,1)

    data_fin = {}

    data_fin['Mean'] = meanDegree
    data_fin['Var'] = varianceDegree
    data_fin['Skew'] = skewDegree
    data_fin['Kurtosis'] = kurtDegree

    #data_fin['Mean_Clos'] = meanDegree_c
    #data_fin['Var_Clos'] = varianceDegree_c
    #data_fin['Skew_Clos'] = skewDegree_c
    #data_fin['Kurtosis_Clos'] = kurtDegree_c


    data = np.loadtxt(file_nino)
    nino = data[:,1]
    time = data[:,0]
    data_fin["ElNino"] = nino
    data_fin["date_time"] = time
    return data_fin


def read_wind_burst(file_name,sep = '\t'):
    dic = {}
    dic['date_time'] = np.array([])
    dic['Sd'] = np.array([])
    dic['wind'] = np.array([])
    data = csv.reader(open(file_name),delimiter=sep, quotechar='|')
    for row in data:
        n = 0
        for j in row:
            if n == 0:
                dic['date_time'] = np.append(dic['date_time'],float(j))
                n = 1
            elif n == 1:
                dic['Sd'] = np.append(dic['Sd'],float(j))
                n = 2
            else:
                dic['wind'] = np.append(dic['wind'],float(j))
    return dic



def read_csv(file_name, sep='\t', head_flag=True, header = []):
    data = csv.reader(open(file_name), delimiter=sep)
    dic = {}
    if head_flag == False:
        for k in header:
            dic[k] = np.array([])

    i=0
    for row in data:
        if head_flag == True and i == 0:
            header = []
            i=1
            for j,value in enumerate(row):
                dic[value] = np.array([])
                header.append(value)
        else:
            for j,value in enumerate(row):
                dic[header[j]] = np.append(dic[header[j]],value)
    return dic





########################### WRITING ###################################


def arff_file(data,attributes,relation,description,output_dir="./",filename="tmp"):
    """
    Writes input data file for weka methods. They are written in weka arff style
    :param data: The dictionary with the dataset
    :param attributes: The attribute pairs telling which intance is numeric/nominal in weka style
    :param relation: Just a description statement
    :param description: Just a description statement
    :param output_dir: The location in which we want to write the file
    :param filename: The name of the file we want to write (without extension, which is arff by definition)
    :return:
    """
    x = []
    for k in attributes:
        x.append(k[0])
    data_write = {}
    data_write['data'] = manip.dic_to_list(data,order=x)[1:]
    data_write['attributes'] = [tuple(l) for l in attributes]
    data_write['relation'] = unicode(relation)
    data_write['description'] = unicode(description)
    data_final = arf.dumps(data_write)
    #print data_final
    fil = open(output_dir + filename + '.arff', "w")
    fil.write(data_final)
    fil.close()

    return None

def csv_file(data,output_dir,filename,order = [],head = True):
    """
    Writes a csv file with the dataset (adding the separator)
    :param data: The dictionary containing all instances with respective outputs
    :param output_dir: The location in which we want to write the file
    :param filename:  The name of the file we want to write
    :param order:  The order in which we want the features to be written. The last key is the output
    :param head: True if we want the header to be written otherwise False
    :return: returns nothing (maybe we can return a False/True for Fail/Success flag)
    """
    with open(output_dir + filename + '.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(manip.dic_to_list(data,order,head),)
    return None

def gp_file(data,filename,output_dir='',order = [],head = False):
    """
    Writes a file readable for GP algorithm with ecj
    :param data: The dictionary containing all instances with respective outputs
    :param filename: The name of the file we want to write
    :param output_dir: The location in which we want to write the file
    :param order: The order in which we want the features to be written. The last key is the output
    :param head: True if we want the header to be written otherwise False
    :return: returns nothing (maybe we can return a False/True for Fail/Success flag)
    """
    f = open(output_dir + filename + '.csv', 'w')
    f.write(str(len(order)-1) + '\n')
    write = csv.writer(f)
    write.writerows(manip.dic_to_list(data,order,head),)
    f.closed

    return None

