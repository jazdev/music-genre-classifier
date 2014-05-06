import sys

###################################################
#    Modify these variables as per need
###################################################

# Directory where the music dataset is located (GTZAN dataset)
GENRE_DIR = "/home/jaz/Desktop/genre-project/genres_dataset"

# Directory where the test music is located
TEST_DIR = "/home/jaz/Desktop/genre-project/genres_test_set"

# All the available genres
GENRE_LIST = [ "blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

# Working with these genres
#GENRE_LIST = [ "blues","jazz","metal","pop","rock"]

if GENRE_DIR is None or GENRE_DIR is "":
    print "Please set GENRE_DIR in config.py"
    sys.exit(1)

elif TEST_DIR is None or TEST_DIR is "":
    print "Please set TEST_DIR in config.py" 
    sys.exit(1)    

elif GENRE_LIST is None or len(GENRE_LIST)==0:
    print "Please set GENRE_LIST in config.py" 
    sys.exit(1)

else:
    print "Variables defined in config.py \n"
    print "GENRE_DIR ==> ", GENRE_DIR
    print "TEST_DIR ==> ", TEST_DIR
    print "GENRE_LIST ==> "," || ".join(x for x in GENRE_LIST)
