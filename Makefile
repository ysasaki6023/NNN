all:NNN
#opt = -O3 -g -pg -shared -Wl,-soname,pynnn.so -lboost_python -lpython2.7 -fPIC
#opt = -O3 -g -pg -shared -lpython2.7 -fPIC
#opt = -O3 -shared -lpython2.7 -fPIC
#opt = -O3 -lpython2.7
#opt = -dynamiclib `python-config --cflags` `python-config --ldflags`
opt = -dynamiclib `/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin/python-config --cflags` `/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin/python-config --ldflags`
opt_end = -fPIC -shared
CXXFLAGS = -I/usr/include/python2.7 -I/Library/Python/2.7/site-packages/numpy/core/include/
LDLIBPATH = 

RNN: nnn.o
	g++ $(opt) $(CXXFLAGS) $(LDLIBPATH) -o RNN.so nnn.o $(opt_end)

nnn.o: main.cxx
	g++ $(opt) $(CXXFLAGS) $(LDLIBPATH) -c main.cxx -o nnn.o $(opt_end)

nnn: nnn.o
	g++ $(opt) $(CXXFLAGS) $(LDLIBPATH) -o nnn main.cxx


clean:
	rm -rf *.o *.so
