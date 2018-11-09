SRCS=main.c
OBJS=$(SRCS:.c=.o)
BIN=app.out
ARCH=compute_32

CC=gcc
CFLAGS=-Wall -Wextra -Werror -std=c99
CC_INCLUDE=-I/usr/local/cuda/include/
LD_PATHS=-L/usr/local/cuda/lib64/
LDFLAGS=-lcublas -lcuda -lcudart


all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(OBJS) $(LD_PATHS) $(LDFLAGS) -o $(BIN)

%.o: %.c
	$(CC) -c $< $(CFLAGS) $(CC_INCLUDE) -o $@

clean:
	rm -f $(OBJS) $(BIN)
