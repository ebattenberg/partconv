import tune
import wave
import sys

def numAudioFrames(file):
    wav = wave.open(file,'r');
    frames = wav.getnframes()
    wav.close()
    return frames

#-------- script start

if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 64

if len(sys.argv) > 2:
    if sys.argv[2].isdigit():
        L = int(sys.argv[2])
    else:
        impulse = sys.argv[2]
        L = numAudioFrames(impulse)
else:
    impulse = '../reverbs/impulse.wav'
    L = numAudioFrames(impulse)

tune.init(N,L)
#tune.pc.verbosity = 1
tune.type = 'mean'
print 'Tuning for N = %u, L = %u\n' % (N,L)
tune.time([],0)

print 'Times for N = %u, L = %u\n' % (N,L)
tune.best(3)

keys = tune.total.keys()
keys.sort()

values = tune.total.values()
values.sort()

#t0 = find_key(tune.total, values[0])hk
#t0 = tune.profile(values[0])






