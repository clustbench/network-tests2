INCLUDE_DIR=include
OBJ_DIR=obj
SOURCE_DIR=src
OUTPUT_DIR=bin
EXECUTABLE=split

CC=gcc
CFLAGS=-c -Wall -I(INCLUDE_DIR)

SOURCES := $(wildcard $(SOURCE_DIR)/*.c)
OBJECTS := $(patsubst $(SOURCE_DIR)/%.c, $(OBJ_DIR)/%.o, $(SOURCES))

all: $(OBJECTS)
	mkdir -p $(OUTPUT_DIR)
	$(CC) $^ -o $(OUTPUT_DIR)/$(EXECUTABLE)

$(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.c
	mkdir -p $(OBJ_DIR)
	$(CC) -I$(INCLUDE_DIR) -c $< -o $@

	
clean:
	rm -rf $(OBJ_DIR)
	rm -rf $(OUTPUT_DIR)
