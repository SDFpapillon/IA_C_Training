# Ia_c_training — build
#
#   make                 build the static library (release)
#   make BUILD=debug     build with debug info and sanitizer-friendly flags
#   make test            build and run all tests/test_*.c
#   make examples        build all examples/*.c
#   make clean           remove build/
#
# Works with GCC/Clang on Linux and MinGW (MSYS2) on Windows.

BUILD   ?= release

# make predefines CC=cc, which does not exist on MinGW: default to gcc
# unless the user set CC explicitly (e.g. make CC=clang).
ifeq ($(origin CC),default)
  CC := gcc
endif

CSTD     := -std=c99
WARNINGS := -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wstrict-prototypes
CPPFLAGS := -Iinclude

ifeq ($(BUILD),debug)
  OPTFLAGS := -O0 -g
else
  OPTFLAGS := -O2 -DNDEBUG
endif

CFLAGS  := $(CSTD) $(WARNINGS) $(OPTFLAGS) $(EXTRA_CFLAGS)
LDLIBS  := -lm

ifeq ($(OS),Windows_NT)
  EXE := .exe
else
  EXE :=
endif

BUILD_DIR := build/$(BUILD)
LIB       := $(BUILD_DIR)/libiactraining.a

SRCS     := $(wildcard src/*.c)
OBJS     := $(patsubst src/%.c,$(BUILD_DIR)/obj/%.o,$(SRCS))

TEST_SRCS := $(wildcard tests/test_*.c)
TEST_BINS := $(patsubst tests/%.c,$(BUILD_DIR)/tests/%$(EXE),$(TEST_SRCS))

EXAMPLE_SRCS := $(wildcard examples/*.c)
EXAMPLE_BINS := $(patsubst examples/%.c,$(BUILD_DIR)/examples/%$(EXE),$(EXAMPLE_SRCS))

.PHONY: all test examples clean

all: $(LIB)

$(LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(BUILD_DIR)/obj/%.o: src/%.c include/nn.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/tests/%$(EXE): tests/%.c tests/test.h $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD_DIR)/examples/%$(EXE): examples/%.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

test: $(TEST_BINS)
	@status=0; \
	for t in $(TEST_BINS); do \
		echo "== $$t"; \
		./$$t || status=1; \
	done; \
	exit $$status

examples: $(EXAMPLE_BINS)

clean:
	rm -rf build
