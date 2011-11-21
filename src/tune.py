
import math
import partconv
import pdb

#global L, N, total, partial, pc, num_benched


def init(inN,inL):
    global L, N, total, partial, pc, num_benched, type
    L = inL
    N = inN
    total = {}
    partial = {}
    num_benched = 0
    type = 'mean'

    pc = partconv.PartConvFilter()
    pc.FS = 44100

def bench(parts,level='total',fs=44100.):
    global pc, type
    M = len(parts)/2
    load = partconv.doubleArray(M+1)
    blocksize = partconv.intArray(M)
    numblocks = partconv.intArray(M)
    for i in range(M):
        blocksize[i] = parts[2*i]
        numblocks[i] = parts[2*i+1]
    print "benching: " + str(parts)
    pc.setup(M,blocksize,numblocks,None,2,'wisdom.wis')
    TRIALS = 1000;
    POLLUTE_EVERY = 3;
    MAX_TIME = 0.1;

    load[M] = 0.0 # holds total load
    for l in range(M):
        t  = pc.bench(l,TRIALS,POLLUTE_EVERY,MAX_TIME)
        load[l] = t*100*fs/blocksize[l]
        load[M] += load[l] 

    pc.cleanup()

    global num_benched
    num_benched += 1
    if level == 'all':
        output = []
        for i in range(M+1):
            output.append(load[i])
        return output
    elif level == 'total':
        return load[M]
    else:
        print 'invalid benching parameter'
        return None
        

def best(num=3):
    # fully profiles the 'num' best times stored in 'total'
    global total
    times = total.keys()
    times.sort()
    print "Best times: "
    for i in range(min(num,len(times))):
        print "%4.3f" % times[i] + ": " + str(total[times[i]])
        load = bench(total[times[i]],'all')
        s = ', '.join("%4.3f" % j for j in load)
        print s + "\n"

    values = total.values()
    values.sort()
    print "Gardner:"
    profile(values[0])
    print "Uniform:"
    profile(values[-1])

def profile(parts):
    # nicely prints the output from bench()
    t = find_key(total,parts) 
    print "%4.3f" % t + ": " + str(parts)
    load = bench(parts,'all')
    s = ', '.join("%4.3f" % j for j in load)
    print s + "\n"

    return load

def find_key(dic, val):
    """return the key of dictionary dic given the value"""
    return [k for k, v in dic.iteritems() if v == val][0]

def time(prev,ptr):
    global partial, total, L, N
    rem = L - ptr
    if ptr == 0:
        B = N
    else:
        B = ptr/2 + N
    
    #divide up remainder using this block size
    numB = math.ceil(float(rem)/B)
    curr = [int(B),int(numB)]
    parts = prev + curr

    # store partial timing
    if len(prev) > 0:
        prev_str = ''.join(str(i)+' ' for i in prev)
        if prev_str in partial:
            prev_time = partial[prev_str]
            print "RECALLED " + str(prev) 
        else:
            prev_time = bench(prev)
            partial[prev_str] = prev_time

    curr_str = ''.join(str(i)+' ' for i in curr)
    if curr_str in partial:
        curr_time = partial[curr_str]
        print "RECALLED " + str(curr)
    else:
        curr_time = bench(curr)
        partial[curr_str] = curr_time

    print "FINISHED: " + str(parts) + "\n"
    if len(prev) > 0:
        total[prev_time + curr_time] = parts
    else:
        total[curr_time] = parts

    # recurse to further divisions
    numB = 2
    while numB*B < rem/4:
        curr = [int(B),int(numB)]
        parts = prev + curr

        print "partial: " + str(parts)
        


        curr_str = ''.join(str(i)+' ' for i in curr)
        if curr_str in partial:
            curr_time = partial[curr_str]
            print "RECALLED " + str(curr)
        else:
            curr_time = bench(curr)
            partial[curr_str] = curr_time


        if len(prev) > 0:
            partial[prev_str + curr_str] = prev_time + curr_time
        else:
            partial[curr_str] = curr_time


        time(parts,ptr+numB*B)
        numB = numB*2 + 2



