
rm dist/*
# gcc -c main.c -g -std=c99 -o dist/main.o
# gcc -c model/my_first_model.c -g -std=c99 -o dist/my_first_model.o
# gcc -c model/mnist.c -g -std=c99 -o dist/mnist.o
# gcc -c model/mnist_hinge.c -g -std=c99 -o dist/mnist_hinge.o
# gcc -c model/mnist_nn.c -g -std=c99 -o dist/mnist_nn.o
gcc -c model/cifar_unet.c -g -std=c99 -o dist/cifar_unet.o
gcc -c lib/matrix.c -g -std=c99 -o dist/matrix.o
gcc -c lib/csv.c -g -std=c99 -o dist/csv.o
# gcc -c lib/layer.c -g -std=c99 -o dist/layer.o
# gcc -c lib/mnist_csv2.c -g -std=c99 -o dist/mnist_csv2.o
gcc -c lib/bmp.c -g -std=c99 -o dist/bmp.o
gcc -c lib/cifar10.c -g -std=c99 -o dist/cifar10.o
gcc dist/* -g -std=c99 -o dist/main -lm
