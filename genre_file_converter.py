import os
from pydub import AudioSegment
import timeit

start = timeit.default_timer()

rootdir = '/home/jaz/Desktop/genre-project/genres_test_set'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = subdir+'/'+file
        if path.endswith("mp3"):
            song = AudioSegment.from_file(path,"mp3")
            song = song[:30000]
            song.export(path[:-3]+"wav",format='wav')

        
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = subdir+'/'+file
        if not path.endswith("wav"):
            os.remove(path)

stop = timeit.default_timer()

print "Conversion time = ", (stop - start) 
