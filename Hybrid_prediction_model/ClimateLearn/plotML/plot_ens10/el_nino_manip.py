
__author__ = 'ruggero'



import string
import numpy as np
import el_nino_io as io

def join_data_network(d_net1,d_net2):
    """
    joins two dictionaries with common or not common keys. Bugged (working for specific data sets: Armin)
    :param d_net1: The first dictionary
    :param d_net2: The second dictionary
    :return: The merged dictionary
    """

    new_dic = {}
    keys = key_union(d_net1.keys(),d_net2.keys())
    for k in keys:
        if is_in_list(k,d_net1.keys()):
            new_dic[k] = d_net1[k]
        else:
            new_dic[k] = d_net2[k]
    return new_dic

def join_data_elnino(d_net,d_nino):
    """
    Joins network dictionaries and el nino dictionary. Works if elnino
    dictionary have data less sampled that network and to complete it
    performs linear interpolation between sampling (maybe generalize type of interpolation)

    :param d_net: the network (Armin) dataset
    :param d_nino: the Elnino3.4 data set
    :return: the merged dictionary
    """

    new_dic = {}
    for k in d_net.keys():
        new_dic[k] = np.array([])
    for k in new_dic.keys():
        new_dic[k] = d_net[k]
    new_dic['ElNino'] = np.array([])
    n = 0
    length = len(d_nino['date_time'])
    for d in new_dic['date_time']:
        if d < d_nino['date_time'][0]:
            new_dic['ElNino'] = np.append(new_dic['ElNino'],d_nino['ElNino'][0])
        elif d > d_nino['date_time'][length - 1]:
            new_dic['ElNino'] = np.append(new_dic['ElNino'],d_nino['ElNino'][length - 1])
        else:
            while not(d >= d_nino['date_time'][n])&(d < d_nino['date_time'][n+1]):
                n = n + 1
            a = d_nino['ElNino'][n]
            a += (d_nino['ElNino'][n+1]-d_nino['ElNino'][n])/(d_nino['date_time'][n+1]-d_nino['date_time'][n])*(d-d_nino['date_time'][n])
            new_dic['ElNino'] = np.append(new_dic['ElNino'],a)
    return new_dic




def join_wind_elnino(d_wind,d_nino):
    inter = key_inter(d_wind['date_time'],d_nino['date_time'])
    key_in = key_union(d_wind.keys(),d_nino.keys())
    print
    new_dic = {}
    for k in key_in:
        new_dic[k] = np.array([])

    i = 0
    j = 0
    new_keys = np.array([])
    for k in key_in:
        if k != 'date_time':
            new_keys = np.append(new_keys,k)



    for d in inter:
        new_dic['date_time'] = np.append(new_dic['date_time'],d)
        while True:
            if d == d_wind['date_time'][i]:
                break
            else:
                i = i + 1
        while True:
            if d == d_nino['date_time'][j]:
                break
            else:
                j = j + 1
        for k in new_keys:
            if is_in_list(k,d_wind.keys()):
                new_dic[k] = np.append(new_dic[k],d_wind[k][i])
            else:
                new_dic[k] = np.append(new_dic[k],d_nino[k][j])
        i = i + 1
        j = j + 1

    return new_dic


def dic_to_list(dic,order = [],head = True):
    """
    Brings a dictionary into a list form for csv printing.
    :param dic: the dictionary
    :param order: the list is ordered according to the vector...
    :param head: True if header needs to be written
    :return: the list of lists
    """
    new_list = []
    header = []
    if order == []:
        for k in dic.keys():
            header.append(k)
    else:
        for k in order:
            header.append(k)
    if head:
        new_list.append(header)

    for i in range(0,len(dic[dic.keys()[0]])):
        lis_part = []
        for k in header:
            lis_part.append(dic[k][i])
        new_list.append(lis_part)
    return new_list

def el_nino_weka_class(dic,classify,t_0,delta_t,tau):
    """
    Prepares a data set used for classification problems according to a classify list
    :param dic: The dictionary of the data set
    :param classify: The classification list
    :param t_0: The initial time
    :param delta_t: The delta t parameter = 0 if the initial time is the initial available one in the data set
    :param tau: The time ahead of prediction
    :return: a dictionary with fully manageable instances for classification
    """
    if t_0 < dic['date_time'][0]:
        t_0 = dic['date_time'][0]
        delta_t = 0
        #print 'delta_t = 0  imposed'
    if t_0-delta_t < dic['date_time'][0]:
        delta_t = t_0 - dic['date_time'][0]
    

    for i in range(0,len(dic['date_time'])):
        if dic['date_time'][i] >= t_0:
            n_init = i
            break
            
    n_tau = -1
    for i in range (n_init,len(dic['date_time'])):
        if dic['date_time'][i] >= dic['date_time'][n_init] + tau:
            n_tau = i
            break
    if n_tau == -1:
        print 'delay too large to build consistent training set. Exiting!'
        exit(1)

    n_delta = -1
    i = n_init 
    while i >= 0:
        if dic['date_time'][n_init] - dic['date_time'][i] >= delta_t:
            n_delta = i
            break
        i = i - 1 
    if n_delta == -1:
        print 'delta_t too large to build set. Exiting!'
        exit(1)


    keys = dic.keys()
    keys.remove('date_time')
    keys.remove('ElNino')

    dic_nn = {}
    dic_nn['t0'] = np.array([])
    dic_nn['t0-deltat'] = np.array([])
    for i in range(0,n_init - n_delta + 1):
        for j in range(0,len(keys)):
            feat = keys[j] + '_' + str(i)
            dic_nn[feat] = np.array([])

    for i in range(0,n_init - n_delta + 1):
        feat = 'ElNino_' + str(i)
        dic_nn[feat] = np.array([])
    dic_nn['Event'] = np.array([])

    
    header = np.array([])
    for k in keys:
        header = np.append(header,k)
    
    
    n_train = len(dic['date_time']) - 1 - n_tau
    assert n_delta >= 0
    for m in range(1,n_train+1):
        dic_nn['t0'] = np.append(dic_nn['t0'],dic['date_time'][n_init + m - 1])
        dic_nn['t0-deltat'] = np.append(dic_nn['t0-deltat'],dic['date_time'][n_delta + m - 1])
        for i in range(0,n_init - n_delta + 1):
            feat = 'ElNino_' + str(i)
            dic_nn[feat] = np.append(dic_nn[feat],dic['ElNino'][n_delta + i + m - 1])
        dic_nn['Event'] = np.append(dic_nn['Event'],classify[n_tau + m - 1])
        for i in range(0,n_init - n_delta + 1):
            for j in range(0,len(keys)):
                feat = keys[j] + '_' + str(i)
                dic_nn[feat] = np.append(dic_nn[feat],dic[keys[j]][n_delta + i + m - 1])

    return dic_nn

def el_nino_weka_regr(dic,t_0,delta_t,tau):
    """
    Prepares a data set used for regression problems
    :param dic: The input dictioanry
    :param t_0: the initial time of the instances
    :param delta_t: the paramter deltat (= 0 if initial time before or equal to first available one)
    :param tau: the time ahead of prediction
    :return: the dictionary of the instances
    """
    if t_0 < dic['date_time'][0]:
        t_0 = dic['date_time'][0]
        delta_t = 0
        #print 'delta_t = 0  imposed'
    if t_0-delta_t < dic['date_time'][0]:
        delta_t = t_0 - dic['date_time'][0]


    for i in range(0,len(dic['date_time'])):
        if dic['date_time'][i] >= t_0:
            n_init = i
            break

    n_tau = -1
    for i in range (n_init,len(dic['date_time'])):
        if dic['date_time'][i] >= dic['date_time'][n_init] + tau:
            n_tau = i
            break
    if n_tau == -1:
        print 'delay too large to build consistent training set. Exiting!'
        exit(1)

    n_delta = -1
    i = n_init
    while i >= 0:
        if dic['date_time'][n_init] - dic['date_time'][i] >= delta_t:
            n_delta = i
            break
        i = i - 1
    if n_delta == -1:
        print 'delta_t too large to build set. Exiting!'
        exit(1)


    keys = dic.keys()
    keys.remove('date_time')
    keys.remove('ElNino')

    dic_nn = {}
    dic_nn['t0'] = np.array([])
    dic_nn['t0-deltat'] = np.array([])
    for i in range(0,n_init - n_delta + 1):
        for j in range(0,len(keys)):
            feat = keys[j] + '_' + str(i)
            dic_nn[feat] = np.array([])

    for i in range(0,n_init - n_delta + 1):
        feat = 'ElNino_' + str(i)
        dic_nn[feat] = np.array([])
    dic_nn['ElNino_tau'] = np.array([])


    header = np.array([])
    for k in keys:
        header = np.append(header,k)


    n_train = len(dic['date_time']) - 1 - n_tau
    print n_train,n_tau,n_delta,n_init
    assert n_delta >= 0
    for m in range(1,n_train+1):
        dic_nn['t0'] = np.append(dic_nn['t0'],dic['date_time'][n_init + m - 1])
        dic_nn['t0-deltat'] = np.append(dic_nn['t0-deltat'],dic['date_time'][n_delta + m - 1])
        for i in range(0,n_init - n_delta + 1):
            feat = 'ElNino_' + str(i)
            dic_nn[feat] = np.append(dic_nn[feat],dic['ElNino'][n_delta + i + m - 1])
        dic_nn['ElNino_tau'] = np.append(dic_nn['ElNino_tau'],dic['ElNino'][n_tau + m - 1])
        for i in range(0,n_init - n_delta + 1):
            for j in range(0,len(keys)):
                feat = keys[j] + '_' + str(i)
                dic_nn[feat] = np.append(dic_nn[feat],dic[keys[j]][n_delta + i + m - 1])

    return dic_nn

def classify(dic,width = 0.417,threshold = 0.5, nominal = False):
    """
    Returns a list of classified el nino events from a dictionary. Width
    determines the minimum time length (in year) to catch an event, and
    threshold determines the value above which el nino events are considered as such

    :param dic: the dictionary comprising a "ElNino" key list
    :param width: the width to classify positive an event
    :param threshold: to classify positive an event
    :param nominal: if True(False) it provides yes(1) and no(0) in classification
    :return: the classification list (np.array)
    """
    new_list = np.array([])
    end = 0
    k = 0
    while end == 0:
        for i in range(k,len(dic['date_time'])):
            if dic['ElNino'][i] >= threshold:
                nino_init = i
                break
            nino_init = i
        for i in range(nino_init+1,len(dic['date_time'])):
            if dic['ElNino'][i] < threshold:
                nino_fin = i - 1
                break
            nino_fin = i
        if dic['date_time'][nino_fin] - dic['date_time'][nino_init] > width:
            for i in range(k,nino_init):
                if nominal:
                    new_list = np.append(new_list,'no')
                else:
                    new_list = np.append(new_list,int(0.0))
            for i in range(nino_init,nino_fin+1):
                if nominal:
                    new_list = np.append(new_list,'yes')
                else:
                    new_list = np.append(new_list,int(1.0))
        else:
            for i in range(k,nino_fin + 1):
                if nominal:
                    new_list = np.append(new_list,'no')
                else:
                    new_list = np.append(new_list,int(0.0))
        k = nino_fin + 1
        if nino_fin == len(dic['date_time'])-1:
            end = 1
        if nino_init > nino_fin:
            end = 1
            for i in range(k,len(dic['date_time'])):
                if nominal:
                    new_list = np.append(new_list,'no')
                else:
                    new_list = np.append(new_list,int(0.0))
    return new_list

def training_test_sets0(res, p_total = 100, p_train = 70 ,p_test = 30 , name_train = 'train_set' , name_test = 'test_set', dir = '', pop = [] , typ = 'arff'):
    assert p_train + p_test <= 100
    
    # dividing the domain into train, void and test parts
    length = len(res)
    init_train = 0
    fin_train = int(length*float(p_total)/100.0*float(p_train)/100.0)
    init_void = fin_train + 1
    fin_void = int(length*float(p_total)/100.0*float(100.0-p_test)/100.0)
    init_test = fin_void + 1  
    fin_test = int(length*float(p_total)/100.0) - 1  

    return   init_test  
    
def training_test_sets(dic, p_total = 100, p_train = 70 ,p_test = 30 , name_train = 'train_set' , name_test = 'test_set', dir = '', pop = [] , typ = 'arff'):
    """
    From a dictionary of prepared and cleaned instances, in prepares training and test set for weka or in csv form
    :param dic: The dictionary of dataset
    :param p_total: the total percentage of data in the discitonary to use
    :param p_train: the percentage going in training set (temporally ordered)
    :param p_test: the percentage of the test set
    :param name_train: name of produced the training file
    :param name_test: name of produced the test file
    :param dir: the directory where the files are saved
    :param pop: a list containing keys which we want to exclude in classification or regression
    :param typ: the type of file. Arff for weka, csv for other methods
    :return: returns a value associated to the ouptut variable (useful for weka but not needed)
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
    #print new_dic.keys()
    # Brings event as the last key element (both for regression and classification)
    if is_in_list('Event',new_dic.keys()):
        keys = new_dic.keys()
        keys.remove('Event')
        keys.append('Event')
    if is_in_list('ElNino_tau',new_dic.keys()):
        keys = new_dic.keys()
        keys.remove('ElNino_tau')
        keys.append('ElNino_tau')


    p = 0
    # writing the attributes
    attr = []
    for k in keys:
        attr.append([k])
    for i in range(0,len(attr)):
        if attr[i][0] != 'Event':
            attr[i].append('REAL')
        else:
            attr[i].append(['yes','no'])
        if(attr[i][0] == 't0'):
            p = i+1
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

    if typ == 'csv':
        io.csv_file(dic_train,dir,name_train,order=keys)
        io.csv_file(dic_test,dir,name_test,order=keys)
    elif typ == 'arff':
        io.arff_file(dic_train,attr,'ElNino_training',u'',dir,name_train)
        io.arff_file(dic_test,attr,'ElNino_test',u'',dir,name_test)
    elif typ == 'all':
        io.csv_file(dic_train,dir,name_train,order=keys)
        io.csv_file(dic_test,dir,name_test,order=keys)
        io.arff_file(dic_train,attr,'ElNino_training',u'',dir,name_train)
        io.arff_file(dic_test,attr,'ElNino_test',u'',dir,name_test)
    else:
        print 'Not allowed file format. Exiting!'
        exit(1)
    return p

def random_training_test_sets(dic, p_train = 70 , p_test = 30, name_train = 'train_set' , name_test = 'test_set', dir = '', pop = [] , typ = 'arff',seed = 0):
    import random
    length = len(dic[dic.keys()[0]])
    seq = range(0,length)
    random.seed(seed)
    random.shuffle(seq)

    init_train = 0
    fin_train = int(length*float(p_train)/100.0)
    init_test = fin_train + 1
    fin_test = int(length) - 1

    new_dic = dic.copy()
    for k in pop:
        new_dic.pop(k, None)
    if is_in_list('Event',new_dic.keys()):
        keys = new_dic.keys()
        keys.remove('Event')
        keys.append('Event')
    if is_in_list('ElNino_tau',new_dic.keys()):
        keys = new_dic.keys()
        keys.remove('ElNino_tau')
        keys.append('ElNino_tau')

    p = 0
    # writing the attributes
    attr = []
    for k in keys:
        attr.append([k])
    for i in range(0,len(attr)):
        if attr[i][0] != 'Event':
            attr[i].append('REAL')
        else:
            attr[i].append(['yes','no'])
        if(attr[i][0] == 't0'):
            p = i+1

    dic_train = {}
    dic_test = {}

    for k in new_dic.keys():
        dic_train[k] = np.array([])
        dic_test[k] = np.array([])
    for i in range(init_train,fin_train+1):
        for k in new_dic.keys():
            dic_train[k] = np.append(dic_train[k],new_dic[k][seq[i]])
    for i in range(init_test,fin_test+1):
        for k in new_dic.keys():
            dic_test[k] = np.append(dic_test[k],new_dic[k][seq[i]])

    if typ == 'csv':
        io.csv_file(dic_train,dir,name_train,order=keys)
        io.csv_file(dic_test,dir,name_test,order=keys)
    elif typ == 'arff':
        io.arff_file(dic_train,attr,'ElNino_training',u'',dir,name_train)
        io.arff_file(dic_test,attr,'ElNino_test',u'',dir,name_test)
    elif typ == 'all':
        io.csv_file(dic_train,dir,name_train,order=keys)
        io.csv_file(dic_test,dir,name_test,order=keys)
        io.arff_file(dic_train,attr,'ElNino_training',u'',dir,name_train)
        io.arff_file(dic_test,attr,'ElNino_test',u'',dir,name_test)
    else:
        print 'Not allowed file format. Exiting!'
        exit(1)
    return p







def El_nino_set(joined,classified,file_name = 'events.csv',directory = ''):
    """
    Method not needed for now
    To write el_nino data, [date,NINO34,event(yes/no)] into a file in
    a given directory. Data is taken from a dic with network and el nino
    and from a list showing the particular classification (yes/no) for the
    same dates
    :param joined:
    :param classified:
    :param file_name:
    :param directory:
    :return:
    """

    nino = {}
    nino['date_time'] = joined['date_time']
    nino['Index'] = joined['ElNino']
    nino['Event'] = classified
    nino['keys'] = ['date_time','Index','Event']
    io.csv_file(nino,directory,file_name)
    return 0







#######
# To conclude: it should transform float datetime values in proper date_time python values
def El_nino_date_time(data):
    key_datetime = ""
    for k in data.keys():
        try:
            day = data[k][0].day()
            key_datetime = k
            break
        except:
            continue
    assert key_datetime != ""



    assert is_in_list("date_time",data.keys())
    return None

######## Useful functions

def is_in_list(k,lis):
    """
    checks if a given elements is in a list
    :param k:  the element
    :param lis:  the list
    :return: returns a flag
    """
    for kk in lis:
        if k == kk:
            return True
    return False


def key_inter(lis1,lis2):
    inter = np.array([])
    flag = 0
    for k in lis1:
        if is_in_list(k,lis2):
            flag = flag + 1
            inter = np.append(inter,k)
    return inter

def key_union(lis1,lis2):
    union = np.array(lis1[:])
    for k in lis2:
        if is_in_list(k,union):
            continue
        else:
            union = np.append(union,k)
    return union
