# Makefile for libsrcnn
# by Raphael Kim

CPP = gcc
CXX = g++
AR  = ar

SRC_PATH = src
OBJ_PATH = obj
BIN_PATH = lib
TARGET   = libsrcnn.a

SRCS = $(wildcard $(SRC_PATH)/*.cpp)
OBJS = $(SRCS:$(SRC_PATH)/%.cpp=$(OBJ_PATH)/%.o)

CFLAGS  = -mtune=native -fopenmp
CFLAGS += -I$(SRC_PATH)

LFLAGS  = -ffast-math
LFLAGS += -O2

all: prepare $(BIN_PATH)/$(TARGET)

prepare:
	@mkdir -p $(OBJ_PATH)
	@mkdir -p $(BIN_PATH)
	@echo $(SRCS)

clean:
	@rm -rf $(OBJ_PATH)/*.o
	@rm -rf $(BIN_PATH)/$(TARGET)

$(OBJS): $(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	@echo "Compiling $< ..."
	@$(CXX) $(CFLAGS) $(LFLAGS) -c $< -o $@

$(BIN_PATH)/$(TARGET): $(OBJS)
	@echo "Linking $@ ..."
	@$(AR) -q $@ $(OBJ_PATH)/*.o
	@cp -rf $(SRC_PATH)/libsrcnn.h $(BIN_PATH)
