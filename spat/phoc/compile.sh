gcc -c -fPIC -I/private/home/ronghanghu/.conda/envs/dev/include/python3.6m cphoc.c
gcc -shared -o cphoc.so cphoc.o -L/private/home/ronghanghu/.conda/envs/dev/lib -lpython3.6m -lpthread -ldl -lutil -lrt -lm
rm cphoc.o
