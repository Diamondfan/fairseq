
import sys 
import soundfile

wav_path = sys.argv[1]

with open(wav_path, 'r') as rf: 
    eachline = rf.readline()
    while eachline:
        fname = eachline.strip()
        frames = soundfile.info(fname).frames
        print("{}\t{}".format(fname, frames))
        eachline = rf.readline()
