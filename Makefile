all:NNN.so
#for MAC (Path to some programs/libraries need to be adjusted to your environment)
#opt = -dynamiclib `/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin/python-config --cflags` `/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin/python-config --ldflags`
#opt_end = -fPIC -shared
#CXXFLAGS = -I/usr/include/python2.7 -I/Library/Python/2.7/site-packages/numpy/core/include/
#for Linux (Path to some programs/libraries need to be adjusted to your environment)
opt = -O3 -shared -lpython2.7 -fPIC
opt_end = -lpython2.7 -fPIC
CXXFLAGS = -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include
LDLIBPATH = 

NNN.so: nnn.o
	g++ $(opt) $(CXXFLAGS) $(LDLIBPATH) -o NNN.so nnn.o $(opt_end)

nnn.o: NNN.cxx
	g++ $(opt) $(CXXFLAGS) $(LDLIBPATH) -c NNN.cxx -o nnn.o $(opt_end)

nnn: nnn.o
	g++ $(opt) $(CXXFLAGS) $(LDLIBPATH) -o nnn NNN.cxx


clean:
	rm -rf *.o *.so
