import numpy
import time

numpy.__config__.show()

m1 = numpy.random.rand(1000, 987)
m2 = numpy.random.rand(987, 876)
#m1 = numpy.array(m, order='F')
#v = numpy.random.rand(9876)

start = time.time()
for i in range(10):
    numpy.dot(m1, m2)
end = time.time()

print(end - start)